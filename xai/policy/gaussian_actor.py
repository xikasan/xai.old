# coding: utf-8

import numpy as np
import tensorflow as tf
from xai.utils.distribution import GaussianDistribution


class GaussianPolicy(tf.keras.Model):

    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(
            self,
            observation_dim,
            action_dim,
            units=[32, 32],
            max_action=1.,
            hidden_activation="relu",
            name="GaussianPolicy"
    ):
        super().__init__(name=name)

        self._max_action = max_action

        self.l1 = tf.keras.layers.Dense(units[0], activation=hidden_activation, name="L1")
        self.l2 = tf.keras.layers.Dense(units[1], activation=hidden_activation, name="L2")
        self.out_mean = tf.keras.layers.Dense(action_dim, name="out_mean")
        self.out_log_std = tf.keras.layers.Dense(action_dim, name="out_log_std")

    def call(self, states):
        params = self._compute_dist(states)
        actions = GaussianDistribution.sample(params)
        log_ps  = GaussianDistribution.log_likelihood(actions, params)

        actions = actions * self._max_action

        return actions, log_ps, params

    def _compute_dist(self, states):
        feature = self.l1(states)
        feature = self.l2(feature)
        mean = self.out_mean(feature)
        log_std = self.out_log_std(feature)
        log_std = tf.clip_by_value(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return {"mean": mean, "log_std": log_std}

    def compute_log_probs(self, states, actions):
        actions /= self._max_action
        params = self._compute_dist(states)
        log_pis = GaussianDistribution.log_likelihood(actions, params)
        return log_pis


if __name__ == '__main__':
    policy = GaussianPolicy(3, 2)
    state = np.random.rand(5, 3).astype(np.float32)
    action, log_pis, params = policy(state)
    print("action", action)
    print("prob", policy.compute_log_probs(state, action))

