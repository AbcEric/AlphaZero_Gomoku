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

# import threading
from subprocess import *

class Yixin(object):
    """
    Yixin AI player
    """

    def __init__(self):
        self.player = None
        self.first = True
        self.p = Popen('/Program Files/Yixin/engine.exe', stdin=PIPE, stdout=PIPE)

    def set_player_ind(self, p):
        self.player = p

    def send_command(self, command=""):
        if not command.endswith("\r\n"):
            command = command + "\r\n"  # 必须要有，代表一行的命令结束。

        self.p.stdin.write(command.encode('GBK'))  # 编码格式与系统有关？这里若为UTF-8，显示中文不正常。
        self.p.stdin.flush()  # 把输入的命令强制输出
        return

    def get_answer(self):
        # line = self.p.stdout.readline().decode("GBK")
        line = str(self.p.stdout.readline().decode("GBK").strip())
        print(">>> ", line)
        while not line.startswith("MESS"):
            line = str(self.p.stdout.readline().decode("GBK").strip())
            # line = self.p.stdout.readline().decode("GBK").strip()
            print(">>>> ", line)

        if not line.startswith("MESS"):
            x = line.split(",")
            print("x=", x, line)
            return int(x[0]), int(x[1])

    def get_action(self, board):
        try:
            if board.last_move == -1:
                # 先走：
                self.send_command("START 15")
                self.p.stdout.readline()
                line = self.p.stdout.readline()
                print(line.decode("GBK"))
                time.sleep(2)
                self.send_command("INFO timeout_turn 1000")  # 每步思考时间：最长1秒，时间越长水平越高。
                time.sleep(1)
                self.send_command("BEGIN")
                x, y = self.get_answer()
                print("xianzhou: x, y = ", x, y)
                move = board.location_to_move(x, y)

            else:
                print("last_move=", board.last_move)
                x = int(board.last_move/15)
                y = board.last_move % 15
                print("x, y: ", x, y)

                if self.first:
                    print("First start ... ")
                    time.sleep(2)
                    self.send_command("START 15")
                    self.p.stdout.readline()
                    self.p.stdout.readline()
                    self.send_command("INFO timeout_turn 1000")  # 每步思考时间：最长1秒，时间越长水平越高。
                    self.first = False

                self.send_command("TURN %d, %d" % (x, y))
                x, y = self.get_answer()
                print("x,y to move = ", x, y)
                move = board.location_to_move(x, y)

        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Yixin {}".format(self.player)


def run():
    # 要采用：PolicyValueNetNumpy(）
    n = 5
    width, height = 8, 8
    model_file = 'best_policy_8_8_5.model'

    # 要采用：PolicyValueNet(）
    n = 5
    width, height = 15, 15
    model_file = 'current_policy.model'
    # model_file = 'best_policy.model'

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
                                 n_playout=400)

        # 采用MCTS作为对手：n_playout越高，水平越厉害（当n_playout=2000时，每步考虑时间就很长了，每步要5秒以上）
        # MCTS的水平很差！基本要3000以上水平才可以。
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=3000)

        # human player, input your move in the format: 2,3
        human = Human()
        yixin = Yixin()

        # set start_player=0 for human first
        # game.start_play(human, mcts_player, start_player=round(random.random()), is_shown=1)
        game.start_play(human, yixin, start_player=round(random.random()), is_shown=1)

    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    # init_logging()
    run()
