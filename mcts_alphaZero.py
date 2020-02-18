# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

import numpy as np
import copy


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
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
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
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


# 大量的计算力花费在了mcts的select与expand上，policy只占了1/8不到的时间（所以deepmind采用了5000个TPU）, 影响效率的主要因素
# 更新mcts时使用了递归，这是慢的一个主要原因，可以考虑改写为堆栈的方式 ？？

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
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while 1:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    # 这一步很花时间：需要2s左右！调用一次就会执行400次MCTS playout。
    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        # print("get_move_probs ...", self._n_playout, state)
        # print("n_playout=", self._n_playout)            # 800?

        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        # temp=1，返回act_probs概率相近的更多，有更多的选择，当temp=0.1时，集中在少数，基本不再探索不同的走法！
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        # print("prob(temp=1):", act_probs[act_probs>0.1])

        return acts, act_probs

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

    # c_puct ？？
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
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs

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
                    p=0.75*probs + 0.25*np.random.dirichlet(0.2*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                # 如果用训练的模型进行相互对战：由于都是选prob最大的走法，会导致相同的局面双方走法完全相同！
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
