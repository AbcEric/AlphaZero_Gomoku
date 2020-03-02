# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)

@author: Junxiao Song
"""

import numpy as np
import copy
from operator import itemgetter

import logging, time, random
_logger = logging.getLogger(__name__)       # 目的是得到当前文件名

def rollout_policy_fn(board):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # 随机选择：
    # action_probs = np.random.rand(len(board.availables))
    # return zip(board.availables, action_probs)

    board.get_move_target()
    # print("move_target: ", board.move_target, len(board.move_target), len(board.availables))
    # print("states: ", board.states)
    # if (len(board.move_target) > 120):
    #     print("!!!!!!!!!!!!!!!!!!!!!",len(board.move_target), board.states)

    action_probs = np.random.rand(len(board.move_target))
    # print("action_probs=", board.move_target, action_probs)

    return zip(board.move_target, action_probs)


def policy_value_fn(board):
    """
    根据board信息输出不同走法机器获胜概率：
    a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state
    """
    # 概率均匀分布 & score为0：
    # action_probs = np.ones(len(board.availables))/len(board.availables)     # p = 1/可能的落子长度，概率都相同？
    # return zip(board.availables, action_probs), 0                           # 返回：（落子，该落子的概率）和评分0（？）

    # 优化为只选已落子周边：
    board.get_move_target()
    action_probs = np.ones(len(board.move_target))/len(board.move_target)     # p = 1/可能的落子长度，概率都相同？
    return zip(board.move_target, action_probs), 0                           # 返回：（落子，该落子的概率）和评分0（？）


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent           # 当为None时，说明为根节点
        self._children = {}             # 当前节点的子节点: 为（编号，子节点)组成的dict
        self._n_visits = 0              # 节点访问次数
        self._Q = 0                     # 根据随机模拟最终结果对节点的评分
        self._u = 0                     # 根据节点访问次数得到的一个加权值，兼顾访问次数少的节点
        self._P = prior_p               # 添加当前节点的子节点时的输入参数：例如从根节点出发，添加可选落点和其概率p为根节点的子节点，形成节点图。

    def expand(self, action_priors):
        """
            Expand tree by creating new children.
            - action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        x = copy.deepcopy(action_priors)
        _logger.debug("len of action = %d" % len(list(x)))

        for action, prob in action_priors:
            # _logger.debug("action = %d" % action)
            if action not in self._children:
                # 将其添加到子节点：
                self._children[action] = TreeNode(self, prob)

        _logger.debug("self.children = %d" % len(self._children))

    def select(self, c_puct):
        """
        根据UCT算法从children中选择最优应对：
        返回：A tuple of (action, next_node)
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
        """
        UCT算法：_Q + _u*c_puct (对访问次数少的给与一定权重)
        - c_puct: 在(0, inf)之间，控制探索和利用的平衡（包括Q，P和n_visits），P为prior probability？
        """
        # print("c_puct=", c_puct)
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

    def __init__(self, policy_value_fn, c_puct=5, n_playout=1000):
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
        self._root = TreeNode(None, 1.0)                # 从只有一个根节点开始
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        _logger.debug("n_playout = %d" % self._n_playout)

    def display_node(self, node, hints=">>> "):
        # node = self._root
        children = node._children
        for move, child in children.items():
            # print("child = ", child)

            if child._n_visits >= 10:
                print("%s %d: Q=%4.3f u=%4.3f (prior_p=%4.3f n_visits=%d)"
                      % (hints, move, child._Q, child._u, child._P, child._n_visits))
            self.display_node(child, "    "+hints)

    def _playout(self, board):
        """
        单独进行一次模拟：根据最终结果得到评分，每次结果反向传播更新到节点Node的访问次数，价值等保持，board每次都是新的。
        Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while 1:
            # 第一次MCTS模拟时，self._children为空，为叶子节点直接返回退出
            if node.is_leaf():
                _logger.debug("current node is leaf, break ....")
                break

            # 根据UCT算法选择move和对应的子节点：
            move, node = node.select(self._c_puct)
            # 根据move走棋更新board：
            board.do_move(move)

        # 根据board信息输出不同走法及其获胜概率：有哪些地方可落子，然后概率均分（如有10个选点，每个选点概率均为1/10）
        action_probs, _ = self._policy(board)

        # Check for end of game
        end, winner = board.game_end()

        if not end:
            # 向下扩展：输入有哪些地方可以落子（action），将这些可能选点加入到当前节点的子节点children
            node.expand(action_probs)
        _logger.debug("add to current node's children(expand): %d" % len(board.availables))

        # 随机行棋，模拟到棋局结束，根据最终结果返回leaf_value.（待优化）
        leaf_value = self._evaluate_rollout(board)
        _logger.debug("leaf_value = %3.2f" % leaf_value)

        # 反向传播：更新访问次数和节点价值等,递归修改节点的父节点，以及节点自身。
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, board, limit=1000):
        """
        采取随机走棋策略：模拟走棋到结束，如果current_player赢返回1，对手赢返回-1，平局返回0
        应该优化为已走棋子的周围，边上走棋无意义。
        """
        player = board.get_current_player()
        for i in range(limit):
            end, winner = board.game_end()
            if end:
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
            return 1 if winner == player else -1

    def get_move(self, board):
        """
        模拟n_playout次，每次将Board信息单独拷贝，进行一次模拟（到结束），
        state: 当前游戏状态，就是Board信息（注意区分不是Board中的states）

        Return: the selected action
        """
        # print("n_playout=", self._n_playout)
        for n in range(self._n_playout):
            _logger.debug("n = %d" % n)
            board_copy = copy.deepcopy(board)       # 将棋盘状态硬拷贝? 包括：availables，current_player,last_move, states等
            self._playout(board_copy)               # 进行一次模拟

        # 显示节点当前状态：
        self.display_node(self._root)

        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        # 用MCTS_Pure生成模拟数据时：？
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            # 重置根节点：两个对手对战时
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    # 是Player1还是Player2：
    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            # 重新定义根节点：已有的信息作废。（训练时将lastmove指定为新的根节点）
            self.mcts.update_with_move(-1)
            print(board.states)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS_Pure {}".format(self.player)
