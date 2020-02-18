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

Q：
改进：
1. 多线程：可用一个进程负责self-play和training的部分, 另外4个进程只负责self-play的部分”。喂数据应该专门开一个进程（用 Queue 储存和读取），这样 GPU 就不会经常停下来等候。
2. 关于GPU不比CPU快这个问题，可能的原因：一是AlphaZero训练本身就有很大一部分运算是需要在cpu上进行的，频繁的在cpu和gpu之间交换数据本身也会有一定开销。（Deepmind用了很多TPU）
其次，棋盘很小，而且我用的网络本身也很浅，所以网络forward计算这部分运算放到GPU上带来的收益可能都被额外的数据传输开销抵掉了。
3. 评价时采用AI模型，加大n_playout作为对手，同时记录对局过程来复盘, 增加装饰器@print_time记录运行时间；
4. 将evalution的步数记录下来：由于用1000的MCTS模拟，也可采用高质量的下法来训练，主要的瓶颈在生成训练数据。（要确保训练数据不重复！）
5. 用图形连续显示运行结果：避免每次刷新，而且可记录手数！

"""

from __future__ import print_function
import random, time
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
# from policy_value_net import PolicyValueNet                   # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet           # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet        # Tensorflow
from policy_value_net_keras import PolicyValueNet               # Keras
from mytoolkit import init_logging, write_log, print_time

class TrainPipeline():
    def __init__(self, init_model=None, best_model="best_policy.model"):
        # params of the board and the game
        self.board_width = 9                                    # 棋盘大小：6x6
        self.board_height = 9
        self.n_in_row = 5                                       # 在6x6棋盘下4子棋
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 0.09                                # adaptively adjust the learning rate based on KL ？
        self.temp = 1.0                                         # temperature parameter，取值在(0, 1]，控制explorati级别，是否去尝试新走法
        self.n_playout = 400                                    # AI只执行400次模拟
        self.c_puct = 5                                         # ？
        self.buffer_size = 10000
        self.batch_size = 512                                   # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)       # data_buffer是一个deque（双向队列），设定了maxlen，满了之后新进来的就会把最老的挤出去
        self.play_batch_size = 1
        self.epochs = 5                                         # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50                                    # 每50次训练后，评估一下性能
        self.game_batch_num = 2000                              # 训练总次数
        self.best_win_ratio = 0.2                               # 当前的最佳胜率，从0开始
        self.pure_mcts_playout_num = 4000                       # 纯MCTS（电脑对手）：每次从1000次MC模拟开始

        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
            # ADD：用best_policy.model作为对手测评：
            self.policy_value_net_best = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=best_model)

        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
            # ADD:
            self.policy_value_net_best = PolicyValueNet(self.board_width,
                                                   self.board_height)

        # MCTS模拟？
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def show_mcts(self):
        print("Game info: %d*%d*%d" % (self.board_width, self.board_height, self.n_in_row))
        print("pure_mcts_playout_num= ", self.pure_mcts_playout_num)
        print("best_win_ration= ", self.best_win_ratio)
        print("play_batch_size= ", self.play_batch_size)
        print("learn_rate= %4.3f*%4.3f" % (self.learn_rate, self.lr_multiplier))

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

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""

        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]

            # print("self-play data is generated!")
            # input("Press any key ....")
            self.episode_len = len(play_data)                       # 代表模拟该局的总步数，每步包括4个棋盘矩阵+概率矩阵+胜负（1，-1）

            # 数据增强：augment the data
            play_data = self.get_equi_data(play_data)               # 总步数变为8倍：

            self.data_buffer.extend(play_data)                      # data_buffer相应增加：
            # print("data_buffer len=", len(self.data_buffer))

    def policy_update(self):
        """update the policy-value net"""
        # (state, mcts_prob, winner_z)
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        # 可否优化？
        # print("len of mini_batch = ", len(mini_batch), mini_batch)

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            # print("epochs: ", i)
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
                print("early stopping loop(kl is too large) ...", end=" ")
                break

        # adaptively adjust the learning rate：学习率的动态调整
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl={:.5f}, "
               "lr_multiplier={:.3f}, "
               "loss={:.3f}, "
               "entropy={:.3f}, "
               "explained_var_old={:.3f}, "          # 用来看value function的学习情况，小于0说明预测很不准，比较理想的情况是在0～1之间逐渐增大
               "explained_var_new={:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
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
                print("双手互搏进行测评，AI对手(current_policy.model)的n_playout加大一倍 ...")
                pure_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                                 c_puct=self.c_puct,
                                                 n_playout=self.n_playout*3)

            elif player == 2:
                print("双手互搏进行测评，AI对手(best_policy.model) ...")
                pure_mcts_player = MCTSPlayer(self.policy_value_net_best.policy_value_fn,
                                             c_puct=self.c_puct,
                                             n_playout=self.n_playout)
            else:
                print("Unknown player")
                exit(0)
        else:
            print("采用MCTS进行测评，pure_mcts_playout为: ", self.pure_mcts_playout_num)
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

            print("%d-%d : winner is player %d (start from player %d)" % (n_games, i+1, winner, i%2+1))

        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games

        print("num_playouts: %d, win: %d, lose: %d, tie: %d" % (self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))

        return win_ratio

    def gen_selfplay_ata(self, num=500):
        # 多线程并行生成：根据现有模型selfplay，记录步数列表

        pass

    def run(self):
        """run the training pipeline"""
        try:

            for i in range(self.game_batch_num):

                self.collect_selfplay_data(n_games=self.play_batch_size)            # 若有多个：生成多个后再policy_update()

                print("batch %d, episode_len=%02d: " % (i+1, self.episode_len), end=" ")        # episode_len为本局的步数

                # data_buffer为数据增强后的总步数，变为实际步数的8倍：batch_size为512步，data_buffer在不断加大, 是否需要这么大 10000？？。
                if len(self.data_buffer) > self.batch_size:
                    # 策略更新？记录访问状态？
                    loss, entropy = self.policy_update()
                else:
                    print("")           # 数据不够：换行

                # check the performance of the current model, and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))

                    # 缺省下10盘, 用MCTSplayer做对手
                    win_ratio = self.policy_evaluate(n_games=5)

                    # win_ratio = self.policy_evaluate(n_games=5, player=2)   # 用best_policy.model做对手（没变化？？）

                    self.policy_value_net.save_model('./current_policy.model')
                    print("Save to: current_policy.model")

                    # 另外一种方式：每次都用当前的模型！
                    if win_ratio > self.best_win_ratio:
                        print("New best policy, save to best_plicy.model!", win_ratio, self.best_win_ratio)
                        self.best_win_ratio = win_ratio

                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        print("Save model to: best_policy.model")
                        self.policy_value_net_best = self.policy_value_net

                        # 当MCTS被我们训练的AI模型完全打败时，pure MCTS AI就升级到每步使用2000次模拟，以此类推，不断增强，
                        # 而我们训练的AlphaZeroAI模型每一步始终只使用400次模拟
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0                   # 从0开始
                            print("pure_mcts_playout increase to: ", self.pure_mcts_playout_num)

                    # TEST：强制更新为current_policy
                    # else:
                    #     # update the best_policy
                    #     self.policy_value_net.save_model('./best_policy.model')


        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':

    training_pipeline = TrainPipeline(init_model="current_policy.model")
    # training_pipeline = TrainPipeline("current_policy.model", "best_policy_0217.model")

    # 记录最后一次的：lr_multiplier, learn_rate, best_win_ratio等
    training_pipeline.show_mcts()               # 增加过程显示，或者记录模拟过程！
    init_logging()

    # training_pipeline.pure_mcts_playout_num = 2000                # 修改MCTS模拟深度
    # training_pipeline.policy_evaluate(n_games=2, player=0)       # 用MCTS作为测试基准: current_policy.model(400) vs. MCTS(1000) 3000比较好
    # training_pipeline.policy_evaluate(n_games=2, player=1)       # 测试AI模型性能: current_policy.model(400) vs. current_policy.model(800)
    # training_pipeline.policy_evaluate(n_games=3, player=2)       # 测试AI模型性能: current_policy.model(400) vs. best_policy.model(400)
    # exit(0)
    training_pipeline.run()
