# coding: utf-8

import numpy as np
import tensorflow as tf
from .policy import Policy
from ..utils.distribution import CategoricalDistribution


class CategoricalPolicy(Policy):

    def __init__(self, units, dim_action, **kwargs):
        if not "name" in kwargs.keys():
            kwargs["name"] = "categorical_policy"
        super().__init__(units, dim_action, **kwargs)

        self.prob = tf.keras.layers.Dense(self.dim_action, activation="softmax", name="prob")

    def call(self, state):
        feature = self.model(state)
        prob = self.prob(feature)
        return prob

    def get_action(self, state):
        is_single = len(state.shape) == 1
        if is_single:
            state = tf.expand_dims(state, axis=0)
        prob = self(state)
        action = tf.math.argmax(prob, axis=1)
        return action

    def get_noisy_action(self, state):
        is_single = len(state.shape) == 1
        if is_single:
            state = tf.expand_dims(state, axis=0)
        prob = self(state)
        action = CategoricalDistribution.sample(prob)
        action = tf.squeeze(action)
        prob = tf.squeeze(prob, axis=0)
        return action, prob

    def compute_logp(self, state, action):
        pi = self(state)
        action = tf.reshape(action, action.shape[0])
        action = tf.cast(action, tf.int32)
        action = tf.one_hot(action, self.dim_action)
        log_pi = CategoricalDistribution.log_likehood(action, pi)
        log_pi = tf.expand_dims(log_pi, axis=-1)
        return log_pi
