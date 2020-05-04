# coding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from ..utils import update


class QNet(tk.Model):

    def __init__(self, units, dim_action, dim_observation, name="QNet"):
        super().__init__(name=name)

        name += "/"
        self.l1 = tk.layers.Dense(units[0], activation="relu", name=name+"L1")
        self.l2 = tk.layers.Dense(units[1], activation="relu", name=name+"L2")
        self.lo = tk.layers.Dense(dim_action, name=name+"Lo")

        dummy_obs = tk.Input((dim_observation,), dtype=tf.float32)
        self(dummy_obs)

    def call(self, inputs):
        feature = self.l1(inputs)
        feature = self.l2(feature)
        feature = self.lo(feature)
        return feature


class DQN:

    def __init__(
            self,
            units,
            dim_action,
            dim_observation,
            lr=1e-3,
            gamma=0.9,
            epsilon=0.05,
            name="DQN"
    ):
        self.name = name
        self._qnet = QNet(units, dim_action, dim_observation, name=name+"/qnet")
        self._optimizer = tf.optimizers.Adam(learning_rate=lr)
        self.dim_action = dim_action
        self.epsilon = epsilon
        self.gamma = gamma

    def select_action_greedy(self, state):
        assert isinstance(state, np.ndarray)
        assert len(state.shape) == 1

        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self._select_action_body(state).numpy()
        return action

    def select_action(self, state):
        rand = np.random.rand()
        if rand < self.epsilon:
            return np.random.randint(0, self.dim_action)
        return self.select_action_greedy(state)

    @tf.function
    def _select_action_body(self, state):
        action = self._qnet(state)
        action = tf.squeeze(action)
        action = tf.argmax(action)
        return action

    def train(self, batch):
        state = batch.state
        action = batch.action.astype(np.int)
        next_state = batch.next_state
        reward = batch.reward
        done = batch.done.astype(np.float32)
        loss = self._train_body(state, action, next_state, reward, done)
        return loss.numpy()

    @tf.function
    def _train_body(self, state, action, next_state, reward, done):
        with tf.GradientTape() as tape:
            tderror = self.tderror(state, action, next_state, reward, done)
            loss = tf.square(tderror)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
        grad = tape.gradient(loss, self._qnet.trainable_variables)
        self._optimizer.apply_gradients(zip(grad, self._qnet.trainable_variables))
        return loss

    def tderror(self, state, action, next_state, reward, done):
        done = 1 - done
        action = tf.one_hot(action, self.dim_action)
        target_q = reward + self.gamma * done * self._qnet(next_state)
        current_q = self._qnet(state)
        loss = tf.stop_gradient(target_q) - current_q
        loss = action * loss
        return loss

    def model(self):
        return self._qnet


class TargetDQN(DQN):

    def __init__(
            self,
            units,
            dim_action,
            dim_observation,
            lr=1e-3,
            gamma=0.9,
            update_rate=0.01,
            epsilon=0.05,
            name="TargetDQN"
    ):
        self.name = name
        self._qnet = QNet(units, dim_action, dim_observation, name=name+"/qnet")
        self._tnet = QNet(units, dim_action, dim_observation, name=name+"/tnet")
        update.copy(self._qnet, self._tnet)

        self._optimizer = tf.optimizers.Adam(learning_rate=lr)
        self.dim_action = dim_action
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = update_rate

    def train(self, batch):
        loss = super().train(batch)
        update.soft_update(self._qnet, self._tnet, tau=self.tau)
        return loss

    def tderror(self, state, action, next_state, reward, done):
        done = 1 - done
        action = tf.one_hot(action, self.dim_action)
        target_q = reward + self.gamma * done * self._tnet(next_state)
        current_q = self._qnet(state)
        loss = tf.stop_gradient(target_q) - current_q
        loss = action * loss
        return loss

    def model(self):
        return self._tnet
