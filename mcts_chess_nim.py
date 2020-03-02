# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS) - chess move
@author: Eric Li

为nim游戏：为博弈论中的经典模型，是组合游戏的一种（ICG）。这是在mcts_number上的改进

长度15x1的棋盘，有n个棋子放在棋盘上，一格可以放多个棋子，每次向左移动一个棋子，当所有棋子都移动到最左边，且最后一个移动的获胜。

Q:
模拟时如何避免相同的走法？否则次数再多也没有什么意义。

"""
import numpy as np
import copy
from operator import itemgetter

def rollout_policy_fn(board):
    # rollout randomly
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0


class Board(object):
    def __init__(self, position):
        # N*m（m<N)
        self.board_len = 15                     # 棋盘长度
        self.chess_num = 3                      # 棋子个数
        self.chess_position = position          # 棋子位置
        self.states = {}                        # ？不太需要？
        self.player = 1                        # 当前player，只能为1或2

    # 根据当前棋子位置更新可用的棋盘availables：
    def update_availables(self):
        for i in range(self.chess_num):
            pos = self.chess_position[i]
            # print(i, pos)
            delpos = [x for x in self.availables if x >= (i * self.board_len + pos) and x < (i + 1) * self.board_len]
            # print(delpos)
            for x in delpos:
                self.availables.remove(x)

    def init_board(self, start_player=1):
        for x in self.chess_position:
            if x < 0 or x >= self.board_len:
                raise Exception('position %d illegal' % x)
        self.player = start_player
        self.states = {}
        # self.last_move = [0, 0, 0]
        self.availables = list(range(self.board_len * self.chess_num))
        self.update_availables()

    def get_location_to_move(self, location):
        if len(location) != self.chess_num:
            print("location length error: ", location)
            return -1

        move = 0
        for i, x in enumerate(location):
            if x > 0 and x <= self.chess_position[i]:
                move = i*self.board_len + self.chess_position[i] - x

        if move not in range(self.board_len * self.chess_num):
            print(move, " not in availables!")
            return -1

        return move


    def do_move(self, move):
        # 先找到原来的位置, 更新对应的chess_postion:
        row = int(move / self.board_len)
        col = move % self.board_len
        self.chess_position[row] = col

        # 记录行棋轨迹：
        self.states[move] = self.player
        # self.availables.remove(move)
        self.update_availables()

        # 切换当前用户：
        self.player = 1 if self.player == 2 else 2

        # self.last_move = move
        return move

    def game_end(self):
        pos = copy.deepcopy(self.chess_position)
        pos.sort()

        if len(self.availables) == 0:
            # LOSS:
            return True, self.player
        elif len(self.availables) == 1:
            # WIN: 在do_move中已切换用户
            return True, 1 if self.player == 2 else 2
        elif max(pos) >= 2 and min(pos) == 0 and pos[-1] == pos[-2]:
            return True, 1 if self.player == 2 else 2
        else:
            return False, -1


class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            # print(action_s, prob)
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.

            c_puct控制收敛到最大value的速度，值越高，说明更多依赖前一个？
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def display_node(self, node, hints=">>> "):
        children = node._children
        for move, child in children.items():
            # print("child = ", child)

            if child._n_visits >= 1:
                print("%s %s: Q=%4.3f u=%4.3f (prior_p=%4.3f n_visits=%d)"
                      % (hints, move, child._Q, child._u, child._P, child._n_visits))

            # self.display_node(child, "    " + hints)

    def _playout(self, board):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while 1:
            if node.is_leaf():
                break

            # 根据UCT算法选择move和对应的子节点：
            action, node = node.select(self._c_puct)
            # print("action=", action)
            board.do_move(action)

        # 根据board信息得到不同走法及其获胜概率：有哪些地方可落子，然后概率均分（如有10个选点，每个选点概率均为1 / 10）
        action_probs, _ = self._policy(board)

        # 是否考虑子节点？
        end, winner = board.game_end()

        if not end:
            # 向下扩展：输入有哪些地方可以落子（action），将这些可能选点加入到当前节点的子节点children
            node.expand(action_probs)

        # 随机行棋，模拟到棋局结束，根据最终结果返回leaf_value.
        leaf_value = self._evaluate_rollout(board)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, board, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        for i in range(limit):
            end, winner = board.game_end()
            if end:
                # print("states=", board.states)
                break

            action_probs = rollout_policy_fn(board)
            max_action = max(action_probs, key=itemgetter(1))[0]
            board.do_move(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")

        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == board.player else -1

    def get_move(self, board):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        for n in range(self._n_playout):
            board_copy = copy.deepcopy(board)
            self._playout(board_copy)

        # print(self._root._children)

        if len(self._root._children) == 0:
            return -1
        else:
            print(self._root._children.items())
            return max(self._root._children.items(),
                key=lambda act_node: act_node[1]._Q)[0]
                # key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        board.update_availables()
        sensible_moves = board.availables
        print("player%d get action begin, availables=%s" % (board.player, sensible_moves))
        if len(sensible_moves) > 0:
            # 为空的处理？
            move = self.mcts.get_move(board)

            # 显示node信息：
            self.mcts.display_node(self.mcts._root)

            if move != -1:
                print("win prob: %4.3f" % self.mcts._root._children[move]._Q)
                self.mcts.update_with_move(-1)

            return move
        else:
            return -1

    def __str__(self):
        return "MCTS {}".format(self.player)


class Game(object):
    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        pass

    def start_play(self, player1, player2, start_player=0, is_shown=0):
        """start a game between two players"""
        self.board.init_board()

        while True:
            current_player = self.board.player
            player1.set_player_ind(1)
            player2.set_player_ind(2)
            players = {1: player1, 2: player2}

            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)

            if move != -1:
                # 移动棋子：
                self.board.do_move(move)
                print("player%d take: %d  now left: %s" % (current_player, move, board.chess_position))

                # 判断胜负：
                end, winner = board.game_end()
                if end:
                    print("Game is over: player%d WIN! " % winner)
                    exit(0)
            else:
                print("Game is over: player%d loss! " % current_player)
                exit(0)

class Human(object):
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            # board.update_availables()
            print("\nplayer%d get_action begin, now left %s" % (board.player, board.chess_position))
            if sum(board.availables) == 0:
                return -1

            location = input("Please input your choice: ")
            if isinstance(location, str):
                location = [int(x, 10) for x in location.split(",")]

            # print("input location=", location)
            # move = board.do_move(location)
            # print(move)
            # 根据列表获得对应的move：
            move = board.get_location_to_move(location)
            # print("move=", move, location)

        except Exception as e:
            print(str(e))
            move = -1

        if move == -1 or move not in board.availables:
            #
            print("invalid move: ", location, move, board.availables)
            self.get_action(board)

        # print("Human retrun move=", move)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def get_all_win_combine(top=11):
    all = []
    for i in range(1,top):
        for j in range(1,top):
            for k in range(1,top):
                if i^j^k == 0:
                    one = [i, j, k]
                    one.sort()
                    if one not in all:
                        all.append(one)
                        print(one)



# 剪枝很重要：减少许多不必要的模拟，如两堆相同(>1)
if __name__ == '__main__':
    # get_all_win_combine(top=21)

    # num = [0, 7, 2, 9]
    num = [4, 7, 9]
    board = Board(num)
    print("FIRST: ", num)
    board.init_board()
    # print(board.availables)
    # print(board.get_availables())
    game = Game(board)
    player1 = MCTSPlayer(c_puct=1/1.414, n_playout=20000)
    # player2 = MCTSPlayer(n_playout=1000)
    player2 = Human()
    game.start_play(player1, player2)
