# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

import numpy as np
import copy, random

import logging
_logger = logging.getLogger(__name__)       # 目的是得到当前文件名


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


# 定义N叉树的TreeNode数据结构：包含了parent和children属性，以及用于计算UCB值的visit times和quality value（Q），
# Node需要实现增加节点、删除节点等功能，还有需要提供函数判断子节点的个数和是否有空闲的子节点位置。
class TreeNode(object):
    """A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode？(action, next_node)
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """
        Expand：第三个步骤
        action_priors: a list of tuples of actions and their prior probability according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    # 根据各节点的价值Q和访问次数多少决定选择哪一个：访问次数少的会多给一些访问机会，get_value()采用UCB算法。
    def select(self, c_puct):
        """
        Select action among children that gives maximum action value Q + bonus u(P).
        Return: A tuple of (action, next_node)，_children是什么结构？
        - action：即move
        - next_node: 下一节点
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update：和update_recursive为第四个阶段
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
        # 递归更新到根节点：
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """UCB算法：计算节点价值
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """叶子节点：即下面没有子节点，i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


# 大量计算花费在MCTS的select与expand上，Policy只占1/8不到的时间（所以deepmind采用了5000个TPU）, 影响效率的主要因素
# 更新MCTS时使用了递归？这是慢的一个主要原因，可以考虑改写为堆栈的方式 ？？
class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """
        从当前根节点到子节点的一次模拟，一直到叶子节点：得到value后，反向传递！
        Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        注意：当前节点指的并不是具体的着法，而是当前整个棋局，其子节点才是具体的着法
        """
        node = self._root
        while 1:
            if node.is_leaf():
                break

            # c_puct控制给为访问次数少的节点多大权重。
            # action为move（要落子的位置），node为下一节点，以便从从下一节点继续循环（children何时修改？）
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # 用state作为输入参数，采用CNN模型计算针对当前player而言的叶子价值和概率(action, probability)
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        end, winner = state.game_end()

        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:        # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # 反向更新节点的value和访问计数：
        node.update_recursive(-leaf_value)

    # 局面分析：依据当前局面得到步骤可能性，每走一步执行400次MCTS模拟，需2s左右！ ?
    # 1. 采用蒙特卡洛下棋
    # 2. 依据蒙特卡洛树来统计每个动作的访问次数
    # 3. 对访问次数做softmax归一化得到概率。
    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and their corresponding probabilities.
        - state: the current game state
        - temp: temperature parameter in (0, 1] controls the level of exploration
        """

        # 蒙特卡洛下棋:
        for n in range(self._n_playout):                    # 循环模拟_n_playout次, 缺省为400
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)                       # 模拟一次行棋：尝试可能的走法，得到value后反向更新。

        # 依据访问计数来计算可能性: calc the move probabilities based on visit counts at the root node?
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)

        # 实验一下：？？？执行保护NaN？？？
        # act_values = [(act, node._Q)
        #               for act, node in self._root._children.items()]
        # acts, visits = zip(*act_values)


        # 归一化：
        # 当temp=1，返回act_probs概率相近的更多，有更多的选择(概率基本和访问次数成正比?)，
        # 当temp=0.1时，集中在少数，基本不再探索不同的走法！e-10是避免log的底数为0
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        - last_move: 当为-1时，修改当前节点为根节点root
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """ AI player based on MCTS """

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    # 关键步骤：
    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables

        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            # 得到当前局面的全部可能"下法"acts和推荐概率probs: 根据n_visits来判断prob，没有什么道理，训练没有什么意义!
            acts, probs = self.mcts.get_move_probs(board, temp)

            # ？？？最后一个才是？？？
            move_probs[list(acts)] = probs

            # 长度为200多，随着落子增加而减少。
            # _logger.info("len of probs = %d" % len(probs))

            # MCTS算法思想：模拟时选择Q+u最大的节点，这是exploitation和exploration的平衡，
            # Q对应exploitation，实际模拟过程中表现好的分支，u对应exploration，充分探索访问次数少的分支，发现是否有更好的策略。
            # 正式下棋的时候，不再需要探索，一般我们选择visit次数最多的分支，这种选择方法相对Robust

            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for self-play training)
                # 当棋盘扩大后dirichlet噪声的参数可能需要调整。根据AlphaZero论文里的描述，这个参数一般按照反比于每一步的可行move数量设置，所以棋盘扩大之后这个参数可能需要减小。
                # 这里的0.3可能减小一些（比如到0.1）
                # dirichlet为狄利克雷分布：应用于构建混合模型，以处理高维的聚类和特征赋权等非监督学习问题？
                move = np.random.choice(
                    acts,
                    # p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                    p=0.75*probs + 0.25*np.random.dirichlet(0.1*np.ones(len(probs)))       # 设置随机从acts数组中各元素取中的概率，越小应该概率差异较大。
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                # 如果用训练的模型进行相互对战：由于都是选prob最大的走法，会导致相同的局面双方走法完全相同！
                move = np.random.choice(acts, p=probs)
                # reset the root node：正式对弈时，每走一步将当前节点变为根节点。

                # 如果是第一步棋，随机选择，避免每次都112：
                if len(board.states) == 0:
                    move = random.choice([96, 97, 98, 111, 112, 113, 126, 127, 128])
                    print("move=", move)

                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            _logger.info("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
