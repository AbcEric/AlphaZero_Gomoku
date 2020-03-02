# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS) - Three group numbers
@author: Eric Li

为nim游戏：为博弈论中的经典模型，是组合游戏的一种（ICG）。
- SG定理（Nim）：对于Anti-SG游戏，先手必赢的情况有两种：1.每堆只有一个物品，且SG=0；2.至少一堆数量>1，且SG=0
按二进制异或后为0必胜，为0的情况移动一步后必不为0，再移动一步可再为0.
若x1^x2^x3 =0 则先手必胜，例如：[[1,2,3], [1,4,5], [1,6,7], [1,8,9], [2,4,6],[2,5,7],[2,8,10],[3,4,7],[3,5,6],[3,9,10]]

或者变种：
长度15x1的棋盘，有n个棋子放在棋盘上，一格可以放多个棋子，每次向左移动一个棋子，当所有棋子都移动到最左边，且最后一个移动的获胜。

"""
import numpy as np
import copy
from operator import itemgetter

def rollout_policy_fn(board):
    # rollout randomly
    # print("available=", board.availables, board.current_num)
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0


class Board(object):
    def __init__(self, numlist):
        # N*m（m<N)
        self.current_num = numlist
        self.states = {}
        self.players = [1, 2]                                   # player1 and player2
        self.current_player = 1

    def init_board(self, start_player=0):
        for x in self.current_num:
            if x < 0:
                raise Exception('number cannot be lower than 0')

        self.current_player = self.players[start_player]        # 当前玩家为start player：谁执黑先走，1为玩家，2为对手
        self.states = {}
        self.last_move = [0, 0, 0]

    def get_availables(self):
        self.availables = []
        # for x, num in enumerate(self.current_num):
        self.availables = [[x+1, 0, 0] for x in range(self.current_num[0])] \
               + [[0, x+1, 0] for x in range(self.current_num[1])][:] \
               + [[0, 0, x+1] for x in range(self.current_num[2])]

        return self.availables

    # 将走法list转换为str，用于做children的主键：
    def list2str(self, move_list):
        return '+'.join(list(map(str, move_list)))

    def str2list(self, move_str):
        move_list = move_str.split("+")
        return list(map(int, move_list))

    def do_move(self, move):
        self.states[self.list2str(move)] = self.current_player
        # print("remove: ", move, self.availables)
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.current_num = list(np.array(self.current_num)-np.array(move))
        self.get_availables()
        self.last_move = move
        return move

    def game_end(self):
        # print("current_num=", self.current_num)
        num_list = copy.deepcopy(self.current_num)
        max_num = max(num_list)
        min_num = min(num_list)
        sum_num = sum(num_list)
        num_list.sort()

        # 剩下一个与一个不剩情况相同：事实上可以只要1个即可，只是速度慢。加上后面是减少很多模型，提升速度。
        if sum_num == 1:
            # 当拿到最后1个时输
            return True, 1 if self.current_player == 2 else 2

        elif min_num == 0 and max_num == int(sum_num/2) > 1:
            # 0-X-X(X>1): 如拿到0-2-2, 0-8-8则输
            return True, 1 if self.current_player == 2 else 2
        elif min_num == 1 and max_num % 2 == 1 and max_num-1 in num_list:
            # 1-2n-2n+1: 如拿到1-4-5, 1-8-9则输
            return True, 1 if self.current_player == 2 else 2
        elif num_list in [[1,2,3], [1,4,5], [1,6,7], [1,8,9], [2,4,6],[2,5,7],[2,8,10],[3,4,7],[3,5,6],[3,9,10]]:
            # 2:4:6: 输
            return True, 1 if self.current_player == 2 else 2

        elif sum_num == 0:
            # print("END WIN：winner=", self.current_player)
            return True, self.current_player
        else:
            return False, -1

    def get_current_player(self):
        return self.current_player


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
            action_s = board.list2str(action)
            # print(action_s, prob)
            if action_s not in self._children:
                self._children[action_s] = TreeNode(self, prob)

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

            if child._n_visits >= 5:
                print("%s %s: Q=%4.3f u=%4.3f (prior_p=%4.3f n_visits=%d)"
                      % (hints, move, child._Q, child._u, child._P, child._n_visits))

            self.display_node(child, "    " + hints)

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
            # print("action=", board.str2list(action))
            board.do_move(board.str2list(action))

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

    def _evaluate_rollout(self, board, limit=20000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        player = board.get_current_player()
        for i in range(limit):
            end, winner = board.game_end()
            if end:
                break

            # print("player%d random rollout: current num is %s, choose from %s" % (board.current_player, board.current_num, board.availables))
            action_probs = rollout_policy_fn(board)
            # print(action_probs)
            # if len(board.availables) != 0:
            max_action = max(action_probs, key=itemgetter(1))[0]
            board.do_move(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")

        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, board):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        for n in range(self._n_playout):
            board_copy = copy.deepcopy(board)
            self._playout(board_copy)

        # print(self._root._children.items())
        # 没有可走的：
        if len(self._root._children.items()) == 0:
            return board.list2str([0, 0, 0])
        else:
            return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

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
        board.get_availables()
        sensible_moves = board.availables
        # if len(sensible_moves) > 0:
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            # print("move=", move)
            # self.mcts.display_node(self.mcts._root)
            if board.str2list(move) != [0, 0, 0]:
                # 获得prob:
                print("win prob: ", self.mcts._root._children[move]._Q)

                self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: only 1 left, LOSE!")
            return board.list2str([0, 0, 0])

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
            current_player = self.board.get_current_player()
            p1, p2 = self.board.players
            player1.set_player_ind(p1)
            player2.set_player_ind(p2)
            players = {p1: player1, p2: player2}

            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)

            if board.str2list(move) != [0, 0, 0]:
                print("player%d take: %s  now left: %s" %
                  (current_player, board.str2list(move), np.array(board.current_num) - np.array(board.str2list(move))))

                self.board.do_move(board.str2list(move))
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
            if sum(board.current_num) == 1:
                return board.list2str([0, 0, 0])

            choice = input("Please input your choice: ")
            if isinstance(choice, str):
                move = [int(x, 10) for x in choice.split(",")]

            # print("input location=", location)
            # move = board.do_move(choice)
            # print(move)
            # location_to_move(location)
        except Exception as e:
            move = [0, 0, 0]

        if move == [0, 0, 0] or move not in board.availables:
            #
            print("invalid move: ", choice, move, board.availables)
            self.get_action(board)

        return board.list2str(move)

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
    get_all_win_combine(top=21)

    num = [11, 17, 20]
    board = Board(num)
    print("FIRST: ", num)
    # print(board.get_availables())
    # exit(0)
    game = Game(board)
    player1 = MCTSPlayer(c_puct=5, n_playout=20000)
    # player2 = MCTSPlayer(n_playout=1000)
    player2 = Human()
    game.start_play(player1, player2)
