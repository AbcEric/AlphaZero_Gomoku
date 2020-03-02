# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""
from __future__ import print_function

from mytoolkit import print_time, load_config
import logging.config
# logging设置只能执行一次，要确保最先执行，在其它包含文件之前，否则包含文件会WARNING及以上才会记录。
logging.config.dictConfig(load_config('./conf/train_config.yaml')['train_logging'])
# 目的得到当前程序名，便于定位。
_logger = logging.getLogger(__name__)

import pickle, random, time
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy          # 纯numpy环境下运行

# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
from policy_value_net_keras import PolicyValueNet               # Keras

import threading
from subprocess import *
from queue import Queue

from mcts_yinxin import YixinPlayer, monitor_yixin_response


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]

            print("input location=", location)
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    # 要采用：PolicyValueNetNumpy(）
    # n = 5
    # width, height = 8, 8
    # model_file = 'best_policy_8_8_5.model'

    # 要采用：PolicyValueNet(）
    n = 5
    width, height = 15, 15
    model_file = 'current_policy.model'
    # model_file = 'best_policy.model'

    conf = load_config('./conf/train_config.yaml')

    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        # 采用作者的预训练模型，只支持6x6或8x8：
        if model_file not in ["best_policy.model", "current_policy.model"]:
            try:
                policy_param = pickle.load(open(model_file, 'rb'))
            except:
                policy_param = pickle.load(open(model_file, 'rb'),
                                           encoding='bytes')  # To support python3

            best_policy = PolicyValueNetNumpy(width, height, policy_param)
        else:
            # 使用自己训练的模型：非numpy方式！
            _logger.info("AI model: %s" % model_file)
            best_policy = PolicyValueNet(width, height, model_file)

        # 采用训练的AI模型作为对手：n_playout越大，水平越高，速度明显快很多！
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=600,)
                                 # is_selfplay=1)             # 运行remove要出错！

        # 采用MCTS作为对手：n_playout越高，水平越厉害（当n_playout=2000时，每步考虑时间就很长了，每步要5秒以上）
        mcts_player2 = MCTS_Pure(c_puct=conf["c_puct"], n_playout=conf["pure_mcts_playout_num"])

        # human player, input your move in the format: 2,3
        human = Human()
        game.start_play(human, mcts_player, start_player=round(random.random()), is_shown=1)
        # game.start_play(mcts_player, mcts_player2, start_player=round(random.random()), is_shown=1)

        '''
        # 用Yixin自己对弈：
        
        qResponses1 = Queue()
        p1 = Popen('C:/Program Files/Yixin/engine.exe', stdin=PIPE, stdout=PIPE)
        child_thread1 = threading.Thread(target=monitor_yixin_response, args=(p1, qResponses1))
        qResponses2 = Queue()
        p2 = Popen('C:/Program Files/Yixin/engine2.exe', stdin=PIPE, stdout=PIPE)
        child_thread2 = threading.Thread(target=monitor_yixin_response, args=(p2, qResponses2))

        # 程序在主线程结束后，直接退出了，不管子线程是否运行完。
        child_thread1.setDaemon(True)
        child_thread1.start()
        child_thread2.setDaemon(True)
        child_thread2.start()

        for i in range(100):
            yixin1 = YixinPlayer(p1, qResponses1, timeout=20)
            yixin2 = YixinPlayer(p2, qResponses2, timeout=1000)
            game.start_play(yixin1, yixin2, start_player=round(random.random()), is_shown=0)

        for i in range(100):
            yixin1 = YixinPlayer(p1, qResponses1, timeout=500)
            yixin2 = YixinPlayer(p2, qResponses2, timeout=300)
            game.start_play(yixin1, yixin2, start_player=round(random.random()), is_shown=0)

        for i in range(100):
            yixin1 = YixinPlayer(p1, qResponses1, timeout=40)
            yixin2 = YixinPlayer(p2, qResponses2, timeout=30)
            game.start_play(yixin1, yixin2, start_player=round(random.random()), is_shown=0)
        '''
            # set start_player=0 for human first


    except KeyboardInterrupt:
        print('\n\rquit')

from mcts_pure import policy_value_fn

if __name__ == '__main__':
    # board = Board(width=15, height=15)
    # board.init_board(0)
    #
    # board.states = {112:1}
    # board.get_move_target()
    # print(board.availables)
    # print(board.move_target, len(board.move_target))
    # exit(0)

    # import mcts_alphaZero
    # import numpy as np
    # visits = [0,1,10,11,12,1,0,0,0,1,2,0]
    # temp = 0.01
    # act_probs = mcts_alphaZero.softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
    # print(act_probs)
    # temp = 1
    # act_probs = mcts_alphaZero.softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
    # print(act_probs)

    run()
