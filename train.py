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
3. 评价时采用AI模型，加大n_playout作为对手，同时记录对局过程来复盘；

"""

from __future__ import print_function
import random
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
    def __init__(self, init_model=None):
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
        self.lr_multiplier = 1.0                                # adaptively adjust the learning rate based on KL ？
        self.temp = 1.0                                         # the temperature param ？
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
        self.best_win_ratio = 0.0                               # 当前的最佳胜率，从0开始
        self.pure_mcts_playout_num = 1000                       # 纯MCTS（电脑对手）：每次从1000次模拟开始

        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)

        # MCTS模拟？
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def show_mcts(self):
        print("Game info: {}*{}*{}".format(self.board_width, self.board_height, self.n_in_row))
        print("pure_mcts_playout_num= ", self.pure_mcts_playout_num)
        print("best_win_ration= ", self.best_win_ratio)
        print("learn_rate= ", self.learn_rate)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        # 每种情况有8种表现形式：旋转或镜像，结果相同
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
            self.episode_len = len(play_data)
            # 数据增强：augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

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
                print("kl = ", kl, " early stopping loop ...")
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
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"          # 用来看value function的学习情况，小于0说明预测很不准，比较理想的情况是在0～1之间逐渐增大
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10, self_play=False):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)

        if self_play:
            print("双手互搏进行测评，AI对手的n_playout加大一倍 ...")
            pure_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                             c_puct=self.c_puct,
                                             n_playout=int(self.n_playout*2))
        else:
            print("采用MCTS进行测评，n_playout为: ", self.pure_mcts_playout_num)
            pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1

            print("%d-%d : winner is player %d" % (n_games, i, winner))

        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games

        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))

        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:

            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)

                # TEST：
                # self.collect_selfplay_data(1)

                print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))
                # input("press to cotinue ...")

                # print(len(self.data_buffer), self.batch_size)
                if len(self.data_buffer) > self.batch_size:
                    # 策略更新：
                    # print("data_buffer len=", len(self.data_buffer))
                    loss, entropy = self.policy_update()

                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()          # 缺省下10盘
                    self.policy_value_net.save_model('./current_policy.model')

                    # 另外一种方式：每次都用当前的模型！
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!", win_ratio, self.best_win_ratio)
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')

                        # 当MCTS被我们训练的AI模型完全打败时，pure MCTS AI就升级到每步使用2000次模拟，以此类推，不断增强，
                        # 而我们训练的AlphaZeroAI模型每一步始终只使用400次模拟
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0                   # 从0开始

                    # TEST：强制更新为current_policy
                    else:
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')


        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline("best_policy.model")
    training_pipeline.show_mcts()

    # training_pipeline.policy_evaluate(n_games=5, self_play=True)
    training_pipeline.policy_evaluate(n_games=5, self_play=False)

    training_pipeline.run()
