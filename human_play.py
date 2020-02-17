# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy          # 纯numpy环境下运行

# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
from policy_value_net_keras import PolicyValueNet               # Keras


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
    n = 5
    width, height = 8, 8
    model_file = 'best_policy_8_8_5.model'

    # 要采用：PolicyValueNet(）
    n = 5
    width, height = 9, 9
    # model_file = 'current_policy.model'
    model_file = 'best_policy.model'

    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        if model_file != "best_policy.model":
            try:
                policy_param = pickle.load(open(model_file, 'rb'))
            except:
                policy_param = pickle.load(open(model_file, 'rb'),
                                           encoding='bytes')  # To support python3

            best_policy = PolicyValueNetNumpy(width, height, policy_param)
        else:
            # 使用自己训练的模型：非numpy方式！
            best_policy = PolicyValueNet(width, height, model_file)

        # 采用训练的AI模型作为对手：n_playout越大，水平越高，速度明显快很多！
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=800)
                                 # n_playout=400)

        # 采用MCTS作为对手：n_playout越高，水平越厉害（当n_playout=2000时，每步考虑时间就很长了，每步要5秒以上）
        # MCTS的水平很差！基本要3000以上水平才可以。
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=3000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=1, is_shown=1)

    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
