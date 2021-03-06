# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet with Keras
Tested under Keras 2.0.5 with tensorflow-gpu 1.2.1 as backend

@author: Mingxu Zhang
""" 

from __future__ import print_function

import tensorflow as tf
from tensorflow import keras

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
from keras.utils import np_utils
import numpy as np
import pickle
# print(tf.__version__)
import logging
_logger = logging.getLogger(__name__)       # 目的是得到当前文件名

class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height 
        self.l2_const = 1e-4                    # coef of l2 penalty
        self.create_policy_value_net()   
        self._loss_train_op()

        if model_file:
            _logger.info("Open model file: %s" % model_file)
            net_params = pickle.load(open(model_file, 'rb'))
            self.model.set_weights(net_params)
        
    def create_policy_value_net(self):
        """create the policy value network """
        # 使用了4个二值特征平面作为输入：前两个表示当前player的棋子位置和对手player的棋子位置，有棋子的位置是1，没棋子的位置是0.
        # 第三个表示对手player最近一步的落子位置，也就是整个平面只有一个位置是1，其余全部是0.
        # 第四个平面表示当前player是不是先手player，如果是先手player则整个平面全部为1，否则全部为0.

        # in_x = network = Input((4, self.board_width, self.board_height))
        in_x = x = Input((4, self.board_width, self.board_height))

        # conv layers
        # model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=input_shape))

        # network = Conv2D(filters=32, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        x = Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(self.l2_const))(x)

        # network = Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        x = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(self.l2_const))(x)

        # network = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        x = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(self.l2_const))(x)

        # action policy layers
        # policy_net = Conv2D(filters=4, kernel_size=(1, 1), data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        x = Conv2D(4, (1, 1), activation="relu", kernel_regularizer=l2(self.l2_const))(x)
        # policy_net = Flatten()(policy_net)
        x = Flatten()(x)

        # self.policy_net = Dense(self.board_width*self.board_height, activation="softmax", kernel_regularizer=l2(self.l2_const))(policy_net)

        # 策略网络：判断落子
        self.policy_net = Dense(self.board_width*self.board_height, activation="softmax", kernel_regularizer=l2(self.l2_const))(x)

        # state value layers
        # value_net = Conv2D(filters=2, kernel_size=(1, 1), data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        # value_net = Flatten()(value_net)
        # value_net = Dense(64, kernel_regularizer=l2(self.l2_const))(value_net)
        # self.value_net = Dense(1, activation="tanh", kernel_regularizer=l2(self.l2_const))(value_net)

        x = Conv2D(2, (1, 1), activation="relu", kernel_regularizer=l2(self.l2_const))(in_x)
        x = Flatten()(x)
        x = Dense(64, kernel_regularizer=l2(self.l2_const))(x)

        # 价值网络：判断胜率
        self.value_net = Dense(1, activation="tanh", kernel_regularizer=l2(self.l2_const))(x)

        self.model = Model(in_x, [self.policy_net, self.value_net])

        # self.model.summary()

        def policy_value(state_input):
            state_input_union = np.array(state_input)
            results = self.model.predict_on_batch(state_input_union)
            return results

        self.policy_value = policy_value
        
    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = board.availables
        current_state = board.current_state()

        # print(legal_positions, current_state)

        act_probs, value = self.policy_value(current_state.reshape(-1, 4, self.board_width, self.board_height))
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])

        # print("act_probs=", list(act_probs), value[0][0])
        return act_probs, value[0][0]

    def _loss_train_op(self):
        """
        Three loss terms：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """

        # get the train op   
        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer=opt, loss=losses)

        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        def train_step(state_input, mcts_probs, winner, learning_rate):
            state_input_union = np.array(state_input)
            mcts_probs_union = np.array(mcts_probs)
            winner_union = np.array(winner)
            loss = self.model.evaluate(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            action_probs, _ = self.model.predict_on_batch(state_input_union)
            entropy = self_entropy(action_probs)
            K.set_value(self.model.optimizer.lr, learning_rate)
            self.model.fit(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            return loss[0], entropy
        
        self.train_step = train_step

    def get_policy_param(self):
        net_params = self.model.get_weights()        
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()
        pickle.dump(net_params, open(model_file, 'wb'), protocol=2)
