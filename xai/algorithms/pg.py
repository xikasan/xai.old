# coding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from ..policy.categorical import CategoricalPolicy


class PG(tk.Model):

    EPS = 1e-8

    def __init__(
            self,
            units,
            dim_action,
            dim_state,
            lr=1e-3,
            discount=0.9,
            name="PG"
    ):
        super().__init__(name=name)

        self.policy = CategoricalPolicy(units, dim_action)
        self.lr = tf.Variable(lr, name="lr", dtype=tf.float32)
        self.optimizer = tk.optimizers.Adam(learning_rate=lr)

        self.disocount = discount

    def select_action(self, state):
        state = state.astype(np.float32)
        action = self._select_action_body(state).numpy()
        action = action[0]
        return action

    @tf.function
    def _select_action_body(self, state):
        return self.policy.get_action(state)

    def select_noisy_action(self, state):
        state = state.astype(np.float32)
        action, _ = self._select_noisy_action(state)
        action = action.numpy()
        return action

    @tf.function
    def _select_noisy_action(self, state):
        return self.policy.get_noisy_action(state)

    def train(self, batch):
        s = batch.state
        a = batch.action
        s_ = batch.next_state
        r = batch.reward
        d = batch.done
        q = batch.Q

        loss = self._train_body(s, a, s_, r, d, q)
        return loss.numpy()

    # @tf.function
    def _train_body(self, s, a, s_, r, d, q):
        with tf.GradientTape() as tape:
            log_pi = self.policy.compute_logp(s, a)
            loss = q * log_pi
            loss = -tf.reduce_mean(loss)
        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))
        return loss

    def compute_value(self, rewards, dones):
        shape = rewards.shape
        rewards = np.squeeze(rewards)
        dones = np.squeeze(dones)
        masks = 1 - dones

        Qs = []
        Q = 0
        for r, m in zip(reversed(rewards), reversed(masks)):
            Q = r + self.disocount * m * Q
            Qs.append(Q)

        Qs = np.asarray(Qs[::-1], dtype=rewards.dtype).reshape(shape)
        return Qs
