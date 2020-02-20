# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku（五子棋）

@author: Junxiao Song

参考：
    https://zhuanlan.zhihu.com/p/32089487 （说明）
    https://link.zhihu.com/?target=https%3A//github.com/junxiaosong/AlphaZero_Gomoku  （代码）

对于在6*6的棋盘上下4子棋这种情况，大约通过500~1000局的self-play训练（2小时），就能训练出比较靠谱的AI
对于在8*8的棋盘上下5子棋这种情况，通过大约2000~3000局自我对弈训练（2天）

MCTS：蒙特卡罗树搜索（Monte Carlo Tree Search），是一类树搜索算法的统称，可以较为有效地解决一些探索空间巨大的问题。
要求的条件是zero-sum（零和博弈，能分出胜负）、fully information（信息公开）、determinism（确定性）、sequential（顺序执行）、discrete（操作是离散）
AlphaGo也是基于MCTS算法，但做了很多优化：http://www.algorithmdog.com/alphago-zero-notes 的介绍，替换了经典实现的UCB算法，
使用policy network的输出替换父节点访问次数，使用子节点访问次数作为分母保证exploration，Q值改为从快速走子网络得到的所有叶子节点的均值，
神经网络也是从前代的CNN改为最新的ResNet

备忘：
300次训练后，loss约4.5，entropy约3.8，explain_var约0.04
900次训练后：loss约3.9~4.2,entropy约3.3，explain_var为负？（当去掉-1平局后，explain_var马上上升？）

优化：
1. logging模块采用.yaml配置文件：方便控制是否控制台显示，训练参数配置，使用也更方便等；（OK）
2. 采用sgf人工比赛棋谱，大大降低训练数据生成时间，训练速度提升1个数量级；同时，数据质量也提高；拓展到15x15标准棋盘；（OK）
3. 评价时加大n_playout作为对手，同时记录对局过程来复盘, 增加装饰器@print_time记录运行时间。采用AI模型作为对手每次相同的情况应对完全一致，没有exploration方式选取；（OK）

Q：
待改进：
1. 多线程：可用一个进程负责self-play和training的部分, 另外4个进程只负责self-play的部分”。喂数据应该专门开一个进程（用 Queue 储存和读取），这样 GPU 就不会经常停下来等候。没必要，放Google Colab上试试；
2. 关于GPU不比CPU快这个问题，可能的原因：一是AlphaZero训练本身就有很大一部分运算是需要在cpu上进行的，频繁的在cpu和gpu之间交换数据本身也会有一定开销。（Deepmind用了很多TPU）
其次，棋盘很小，而且我用的网络本身也很浅，所以网络forward计算这部分运算放到GPU上带来的收益可能都被额外的数据传输开销抵掉了。
3. 将evalution的步数记录下来：由于用1000的MCTS模拟，也可采用高质量的下法来训练，主要的瓶颈在生成训练数据。（要确保训练数据不重复！）
4. 用图形连续显示运行结果：避免每次刷新，而且可记录手数！
5. MCTS采用递归性能较低，能否采用队列方式实现？来提升效率，进一步研究MCTS

"""
from __future__ import print_function

from mytoolkit import print_time, load_config
import logging.config
# logging设置只能执行一次，要确保最先执行，在其它包含文件之前，否则包含文件会WARNING及以上才会记录。
logging.config.dictConfig(load_config('./conf/train_config.yaml')['train_logging'])
# 目的得到当前程序名，便于定位。
_logger = logging.getLogger(__name__)

import random, os
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
# from policy_value_net import PolicyValueNet                   # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet           # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet        # Tensorflow
from policy_value_net_keras import PolicyValueNet               # Keras


class TrainPipeline():
    def __init__(self, conf, init_model=None, best_model="best_policy.model"):
        # params of the board and the game
        self.board_width = conf['board_width']                  # 棋盘大小：6x6
        self.board_height = conf['board_height']
        self.n_in_row = conf['n_in_row']                        # 在6x6棋盘下4子棋
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # training params
        self.learn_rate = conf['learn_rate']
        self.lr_multiplier = conf['lr_multiplier']              # 用于动态调整学习率
        self.temp = conf['temp']                                # temperature parameter，取值在(0, 1]，控制explorati级别，是否去尝试新走法
        self.n_playout = conf['n_playout']                      # 缺省AI只执行400次模拟
        self.c_puct = conf['c_puct']                                         # ？
        self.buffer_size = conf['buffer_size']
        self.batch_size = conf['batch_size']                    # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)       # data_buffer是一个deque（双向队列），设定了maxlen，满了之后新进来的就会把最老的挤出去
        self.play_batch_size = conf['play_batch_size']
        self.epochs = conf['epochs']                            # num of train_steps for each update
        self.kl_targ = conf['kl_targ']                          # KL散度
        self.check_freq = conf['check_freq']                    # 每50次训练后，评估一下性能
        self.game_batch_num = conf['game_batch_num']            # 训练总次数
        self.best_win_ratio = 0.2                               # 当前的最佳胜率，从0开始
        self.pure_mcts_playout_num = conf['pure_mcts_playout_num']      # 纯MCTS（电脑对手）：每次从1000次MC模拟开始

        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
            # ADD：用best_policy.model作为对手测评：
            # self.policy_value_net_best = PolicyValueNet(self.board_width,
            #                                        self.board_height,
            #                                        model_file=best_model)

        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
            # ADD:
            # self.policy_value_net_best = PolicyValueNet(self.board_width,
            #                                        self.board_height)

        # 用MCTS作为模拟对手：
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def show_mcts(self):
        _logger.info("Game info: %d*%d*%d" % (self.board_width, self.board_height, self.n_in_row))
        _logger.info("pure_mcts_playout_num= %d" % self.pure_mcts_playout_num)
        _logger.info("best_win_ration= %43f" % self.best_win_ratio)
        _logger.info("play_batch_size= %d" % self.play_batch_size)
        _logger.info("learn_rate= %4.3f*%4.3f" % (self.learn_rate, self.lr_multiplier))

    def get_equi_data(self, play_data):
        # 数据增强：每种情况有8种表现形式，通过旋转或镜像后，结果相同

        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1, train_data=[]):
        # 生成训练数据：两种方式

        # for i in range(n_games):
        for i in range(1):      # 暂不支持一次生成多个
            _logger.debug("use human gomoku, train_data=%s" % train_data)

            winner, play_data = self.game.start_from_train_data(self.mcts_player, train_data, is_shown=0)
            # winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)

            play_data = list(play_data)[:]

            # 代表模拟该局的总步数，play_data返回数据的每一步包括：4个棋盘矩阵+概率矩阵+胜负（1，-1）
            self.episode_len = len(play_data)

            # 数据增强：
            play_data = self.get_equi_data(play_data)               # 总步数变为8倍：
            self.data_buffer.extend(play_data)                      # data_buffer相应增加：

        return

    def policy_update(self):
        # 更新价值网络：评估其预测性能
        # 输入数据为(state, mcts_prob, winner_z)

        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]          # 即为data_buffer的第1列
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)

            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )

            if kl > self.kl_targ * 4:       # early stopping if D_KL diverges badly: >0.08
                _logger.info("early stopping loop(kl=%4.3f is too large) ..." % kl)
                break

            # _logger.info("%d-%d: policy_value_net loss=%4.3f" % (self.epochs, i+1, loss))

        # 学习率的动态调整
        # if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.05:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        # explained_var说明预测情况，比较理想的情况是在0～1之间逐渐增大：
        _logger.info("kl=%4.3f, lr_multiplier=%4.3f, loss=%4.3f, "
                     "entropy=%4.3f, explained_var_old=%4.3f, explained_var_new=%4.3f"
                     % (kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))

        return loss, entropy

    @print_time
    def policy_evaluate(self, n_games=10, player=0):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training

        player: 0(缺省)采用MCTSplayer作为对手，1为current_policy.model, 2为best_policy.model
        n_playout: 适用于MCTSplayer，用作基准（当>2000时，运行速度慢）
        训练用自己做对手有问题！每次应对都一样，不断循环，没有改进和提升 .......
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)

        if player:
            # 或者：与best_model进行PK
            if player == 1:
                _logger.info("双手互搏进行测评，AI对手(current_policy.model)的n_playout加大一倍 ...")
                pure_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                                 c_puct=self.c_puct,
                                                 n_playout=self.n_playout*2)

            elif player == 2:
                _logger.info("双手互搏进行测评，AI对手(best_policy.model) ...")
                pure_mcts_player = MCTSPlayer(self.policy_value_net_best.policy_value_fn,
                                             c_puct=self.c_puct,
                                             n_playout=self.n_playout)
            else:
                _logger.info("Unknown player")
                exit(0)
        else:
            _logger.info("采用MCTS模拟进行测评，pure_mcts_playout为: %d" % self.pure_mcts_playout_num)
            pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)

        # print(type(pure_mcts_player))

        win_cnt = defaultdict(int)

        for i in range(n_games):
            # print(current_mcts_player, pure_mcts_player)
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,           # 轮流先走：
                                          is_shown=0)
            win_cnt[winner] += 1

            _logger.info("%d-%d : winner is player %d (start from player %d)" % (n_games, i+1, winner, i%2+1))

        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games

        _logger.info("num_playouts: %d, win: %d, lose: %d, tie: %d" % (self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))

        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            # 使用人工棋谱：改到外面去，先去掉结果为-1的对局。
            train_data = gen_moves_from_sgf(conf["sgf_dir"])
            random.shuffle(train_data)
            # 暂时去掉平局的对局：
            train_data_without_tie = [[item[0], item[1]] for item in train_data if item[0] != -1]
            # print(train_data_without_tie[0:10], len(train_data_without_tie))

            for i in range(self.game_batch_num):
                # 3-1. 生成训练数据：可一次生成多个，通常play_batch_size为1
                # 既可用训练好的模型来生成数据（步数越多花费的时间越长），也可用棋谱数据来生成（速度快）。
                self.collect_selfplay_data(n_games=self.play_batch_size, train_data=train_data_without_tie[i])
                # episode_len为本局的步数
                _logger.info("batch %d, episode_len=%02d: " % (i+1, self.episode_len))

                # data_buffer为数据增强后的总步数，不断加大, 直到10000（？多大合适？），每次从buffer中随机选取512个来训练。
                if len(self.data_buffer) > self.batch_size:

                    # 3-2. 策略更新: 根据训练数据，更新策略模型的参数，使得loss减少。当棋谱较大时，更新较慢！
                    loss, entropy = self.policy_update()
                    # _logger.info("policy_update end, loss=%4.3f" % loss)

                # 3-3. 定期对模型进行评估：policy_evaluate(), 并保存模型参数（只保存参数，文件小！）
                if (i+1) % self.check_freq == 0:
                    _logger.info("current self-play batch: %d" % (i+1))

                    # 先保存到当前模型：
                    self.policy_value_net.save_model('./current_policy.model')
                    _logger.info("保存当前训练模型到：current_policy.model")

                    # 缺省下10盘, 用MCTSplayer做对手：这里用4次，各自先手2次，后手2次。
                    win_ratio = self.policy_evaluate(n_games=4)
                    # win_ratio = self.policy_evaluate(n_games=5, player=2)   # 用best_policy.model做对手（没变化？？）

                    # 评价后决定是否保存到best_policy模型：
                    if win_ratio > self.best_win_ratio:
                        self.policy_value_net.save_model('./best_policy.model')
                        _logger.info("New best policy(%3.2f>%3.2f), save to best_plicy.model!" % (win_ratio, self.best_win_ratio))
                        # self.policy_value_net_best = self.policy_value_net
                        self.best_win_ratio = win_ratio

                        # 当MCTS被我们训练的AI模型完全打败时，pure MCTS AI就升级到每步使用2000次模拟，以此类推，不断增强，
                        # 而我们训练的AlphaZeroAI模型每一步始终只使用400次模拟
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0                   # 从0开始
                            _logger.info("pure_mcts_playout increase to %d" % self.pure_mcts_playout_num)

        except KeyboardInterrupt:
            _logger.error('quit')

def content_to_order(sequence):
    # 棋谱字母转整型数字
    LETTER_NUM = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
    BIG_LETTER_NUM = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    NUM_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # 棋盘字母位置速查表
    seq_lookup = dict(zip(LETTER_NUM, NUM_LIST))
    num2char_lookup = dict(zip(NUM_LIST, BIG_LETTER_NUM))

    seq_list = sequence.split(';')
    seq_list = [item[2:4] for item in seq_list]
    seq_num_list = [seq_lookup[item[0]]*15+seq_lookup[item[1]] for item in seq_list]
    return seq_list, seq_num_list


def gen_moves_from_sgf(sgf_path, refresh=False):
    sgf_filelist = os.listdir(sgf_path)
    sgf_filelist = [item for item in sgf_filelist if item.endswith('.sgf') and os.path.isfile(os.path.join(sgf_path, item))]
    result = []

    if not refresh:     # 不用重新生成
        try:
            f = open("human_gomoku.txt", "r")

            while 1:
                p = f.readline().strip()
                if len(p) > 0:
                    onegame = eval(p)
                    # print(one, type(one))
                    result.append(onegame)
                else:
                    break
            f.close()
            # print("result=", result[-1])
            return result
        except Exception as e:
            _logger.error("human_gomoku.txt doesn't exist: %s" % str(e))

    _logger.info("generate from %s" % sgf_path)
    fw = open("human_gomoku.txt", "w")
    for file_name in sgf_filelist:
        with open(os.path.join(sgf_path, file_name)) as f:
            p = f.read()            # 只有一行数据
            sequence = p[p.index('SZ[15]')+7:-3]
            seq_num_list = []

            try:
                seq_list, seq_num_list = content_to_order(sequence)

                # 检查棋子是否有重复：
                if len(seq_num_list) != len(list(set(seq_num_list))):
                    _logger.warning("%s: 有重复落子 - %s" % (file_name, seq_num_list))
                    continue
                # _logger.debug("seq_list=%s, seq_num=%s" % (seq_list, seq_num_list))
            except Exception as e:
                _logger.error('file=%s, error:%s' % (file_name, str(e)))
                exit(0)

            if "黑胜" in file_name:
                winner = 1
                if len(seq_num_list) % 2 != 1:
                    _logger.warning("%s: the winner 1 maybe wrong. " % file_name)
            elif "白胜" in file_name:
                winner = 2
                if len(seq_num_list) % 2 != 0:
                    _logger.warning("%s: the winner 2 maybe wrong. " % file_name)
            else:
                winner = -1



            # 检查是否需要copy.deepcopy(xxx)？前面已经重新赋值！
            result.append([winner, seq_num_list])
            fw.write(str([winner, seq_num_list]))
            fw.write("\n")
            # return {'winner': winner, 'seq_list': seq_list, 'seq_num_list': seq_num_list, 'file_name':file_name}

    fw.close()
    return result

if __name__ == '__main__':
    _logger.info("Training is begining ...")
    conf = load_config('./conf/train_config.yaml')

    # 强制重新生成数据：某些是超时判负，某些是先手双三（禁手判负）
    # train_data = gen_moves_from_sgf(conf["sgf_dir"], refresh=True)

    training_pipeline = TrainPipeline(conf, init_model="current_policy.model")
    # training_pipeline = TrainPipeline(conf, init_model=None)      # 首次训练

    # 记录最后一次的：lr_multiplier, learn_rate, best_win_ratio等
    # _logger能否修改配置文件？
    training_pipeline.show_mcts()               # 增加过程显示，或者记录模拟过程！

    # training_pipeline.pure_mcts_playout_num = 2000                # 修改MCTS模拟深度
    # training_pipeline.policy_evaluate(n_games=2, player=0)       # 用MCTS作为测试基准: current_policy.model(400) vs. MCTS(1000) 3000比较好
    # training_pipeline.policy_evaluate(n_games=2, player=1)       # 测试AI模型性能: current_policy.model(400) vs. current_policy.model(800)
    # training_pipeline.policy_evaluate(n_games=3, player=2)       # 测试AI模型性能: current_policy.model(400) vs. best_policy.model(400)
    # exit(0)

    training_pipeline.run()                     # 开始训练
