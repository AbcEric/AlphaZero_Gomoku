# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np
import logging, time, random
_logger = logging.getLogger(__name__)       # 目的是得到当前文件名

class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}                                    # 记录当前棋盘的状态，键是位置，值是棋子。如{112:1, 113:2}，代表112这个位置是Player1已经落子。
        self.n_in_row = int(kwargs.get('n_in_row', 5))      # need how many pieces in a row to win，缺省为5
        self.players = [1, 2]                               # player1 and player2
        self.move_target = []                                    # 模拟走子对象：只走已落子的周围，[+/-4]

    def move_in_rect(self, move, rect):
        row = int(move/self.width)
        col = move % self.width

        if row >= rect[1] and row <= rect[0] and col >= rect[2] and col <= rect[3]:
            # print(row, col, [rect[1], rect[0]], [rect[2], rect[3]])
            return True
        else:
            return False

    def get_move_target(self):
        # 已走位置：
        moves = [[int(x / self.width), x % self.width] for x in self.states.keys()]
        # print("moves=", moves)

        if len(moves) == 0:     # 未走棋
            top, bottom, left, right = 9, 5, 5, 9
        else:
            top = max(np.array(moves)[:, 0])
            bottom = min(np.array(moves)[:, 0])
            left = min(np.array(moves)[:, 1])
            right = max(np.array(moves)[:, 1])

            step = 2
            top = top+step if top <= self.height-step else self.height
            bottom = bottom-step if bottom >= step else 0
            left = left-step if left >= step else 0
            right = right+step if right <= self.width-step else self.width

        # print(top, bottom, left, right)

        self.move_target = [x for x in self.availables if self.move_in_rect(x, [top, bottom, left, right])]

        return

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be less than ', self.n_in_row)

        self.current_player = self.players[start_player]    # 当前玩家为start player：谁执黑先走，1为玩家，2为对手
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))     # 哪些位置还可以走棋
        self.states = {}                                            # -1表示当前位置为空
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state: 4个矩阵(均为height*width)，包括player1 moves, player2 moves, last move and current player(先后手)
        """

        square_state = np.zeros((4, self.width, self.height))

        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0

            # 上一步的落子：indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0

        # 先后还是后手：
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play

        return square_state[:, ::-1, :]

    # 落子：修改board的相关信息
    def do_move(self, move):
        self.states[move] = self.current_player
        # _logger.info("move=%d, current_player=%d, avai=%s" % (move, self.current_player, self.availables))
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row + 2:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""

        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        print('\r\n')
        for i in range(height - 1, -1, -1):         # 倒序：行数从下面开始
            print("%4d" % i, end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print("%2s" % 'X', end=' ')
                    # print("%8s" % '●', end=' ')
                elif p == player2:
                    print("%2s" % 'O', end=' ')
                    # print("%8s" % '○', end=' ')
                else:
                    print("%2s" % '-', end=' ')
            print('')

        print("%4s" % ' ', end='')
        for x in range(width):
            print("%2d" % x, end=' ')
        print('\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        # 人机对战：

        # is_shown = 1
        play_steps = []

        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')

        # start_player为0和1：为0时为player1先走，1为player2（对手）先走：
        if start_player == 1:
            _logger.debug("player %d starts first ... " % (start_player+1))
            # play_steps.append(-1)

        self.board.init_board(start_player)     #
        p1, p2 = self.board.players             # [1, 2]玩家编号
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}    # 区分self.board.players为[1,2]，这里为两个玩家对象

        # print("players=", players)

        if is_shown:
            self.graphic(self.board, player1.player, player2.player)

        _logger.debug("start_player=%d(0为玩家，1为对手) " % (start_player))
        first_step = True

        while True:
            # 注意：current_player中1为玩家，2为对手
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]        # 返回是输入参数的player1还是player2

            # 注意：SGF格式为行X列（aa,ab,ac...），而Yixin为列X行(A1,B1,C1...)，二者均是左下角为0,0，二者为转置关系，要转换为SGF方式：
            # 采用模型与AI对战：双方轮流走棋，对弈步骤都记录在board中。
            move = player_in_turn.get_action(self.board)

            # 如果是player1的第一步: 随机选择，避免每次都112开局。player2的开局不一定112（不再处理）
            # if first_step and start_player == 0:
            #     h = random.randint(4, 10)
            #     w = random.randint(4, 10)
            #     move = self.board.location_to_move([h, w])
            #     _logger.debug("player1先走，第一步随机选择：%d(%d,%d)" % (move, h, w))
            #     first_step = False

            # 由Yinxin AI给出move：
            # move = self.yixin_action(self.board, play_steps)

            play_steps.append(move)
            _logger.debug("player%d: %d" % (current_player, move))

            self.board.do_move(move)

            if is_shown:
                self.graphic(self.board, player1.player, player2.player)

            # 判断是否结束：
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        _logger.debug("Game end：Winner is player%d" % winner)
                    else:
                        _logger.debug("Game end：Tie")

                # 将int列表转换为str列表后，才能用join连接：# players[winner]为"MCTS 1/2"。winner是1或2
                winner_r = winner
                if start_player == 1:
                    # 当player2先走时:
                    if winner == 1:
                        # winner为2，对应player2获胜，由于他先走，此时结果相当于1获胜。
                        winner_r = 2
                    else:
                        winner_r = 1

                _logger.warning("{'steps':[%s],'winner':%d,'start_player':%d}" % (",".join(map(str, play_steps)), winner_r, start_player+1))

                return winner

    def start_from_train_data(self, player, train_data, is_shown=0):
        """
        使用棋谱或预先生产的棋谱，即可节约时间，又可充分利用高质量的棋谱。
        """

        # 获取棋盘数据
        # X_train_str = "40 37 30 36 50 20 60 70 48 56 39 38 21 12 57"
        # X_train = list(map(int, X_train_str.split(' ')))
        # train_data = [1, [12,12,33,44,56]]

        winner = train_data[0]
        X_train = train_data[1]
        data_length = len(X_train)                      # 对弈长度（一盘棋盘数据的长度）

        # _logger.info("X_train=%s seq_len=%d winner=%d" % (X_train, data_length, winner))

        self.board.init_board()
        p1, p2 = self.board.players
        # print('p1: ', p1, '   p2:  ', p2)
        states, mcts_probs, current_players = [], [], []
        # while True:
        for num_index, move in enumerate(X_train):
            probs = [0.000001 for _ in range(self.board.width*self.board.height)]
            probs[move] = 0.99999
            move_probs = np.asarray(probs)

            # print("move=", move, "move_probs=", move_probs, "current_player=", self.board.current_player)
            # print("state=", self.board.current_state())

            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)

            # perform a move
            try:
                self.board.do_move(move)
            except Exception as e:
                print(str(e))
                warning, winner, mcts_probs = 1, None, None
                return warning, winner, mcts_probs

            if is_shown:
                self.graphic(self.board, p1, p2)

            # 重新设置end: 均是黑先，根据对弈长度判断是谁胜利。
            # Q: 看看黑是1还是0，还有和棋胜负如何判断。??
            end, warning = 0, 1
            if num_index + 1 == data_length:
                end = 1
                # winner = data_length % 2

            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0

                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")

                return winner, zip(states, mcts_probs, winners_z)

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        # 下棋过程：互搏
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        moves = []          # 记录对弈步骤
        i = 1

        while True:
            # 得到每一步落子：
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # 判断移动哪一个棋子：
            _logger.debug("%d - move: %d" % (i, move))
            i += 1

            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            moves.append(move)

            # perform a move
            self.board.do_move(move)

            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()

            if end:
                # winner from the perspective of the current player of each state

                winners_z = np.zeros(len(current_players))      # 平局为0
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0

                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")

                # _logger.info("%d-%s" % (winner, moves))

                _logger.warning("{'steps':[%s],'winner':%d}" % (",".join(map(str, moves)), winner))

                # print("states=", states)
                # print("mcts_probs=", mcts_probs)
                # print("winners_z=", winners_z)
                return winner, zip(states, mcts_probs, winners_z)
