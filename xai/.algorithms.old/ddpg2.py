# coding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from ..utils import update


class Critic(tk.Model):

    def __init__(
            self,
            units,
            dim_state,
            dim_action,
            dtype=tf.float32,
            name="critic"
    ):
        super().__init__(name=name)

        name += "/"
        self.l1 = tk.layers.Dense(units[0], activation="relu", name=name+"L1")
        self.l2 = tk.layers.Dense(units[1], activation="relu", name=name+"L2")
        self.lo = tk.layers.Dense(1, name=name+"Lo")

        dummy_action = tk.Input((dim_action,), dtype=dtype)
        dummy_state = tk.Input((dim_state,), dtype=dtype)
        self([dummy_state, dummy_action])

    def call(self, inputs, **kwargs):
        feature = tf.concat(inputs, axis=1)
        feature = self.l1(feature)
        feature = self.l2(feature)
        feature = self.lo(feature)
        return feature


class Policy(tk.Model):

    def __init__(
            self,
            units,
            dim_state,
            dim_action,
            max_action=1.0,
            dtype=tf.float32,
            name="policy"
    ):
        super().__init__(name=name)

        name += "/"
        self.l1 = tk.layers.Dense(units[0],   activation="relu", name=name+"L1")
        self.l2 = tk.layers.Dense(units[0],   activation="relu", name=name+"L2")
        self.lo = tk.layers.Dense(dim_action, activation="tanh", name=name+"Lo")
        self.max_action = max_action

        dummy_state = tk.Input((dim_state,), dtype=dtype)
        self(dummy_state)

    def call(self, inputs, **kwargs):
        feature = self.l1(inputs)
        feature = self.l2(feature)
        feature = self.lo(feature)
        feature = feature * self.max_action
        return feature


class DDPG(tk.Model):

    def __init__(
            self,
            dim_state,
            dim_action,
            policy_units=[400, 300],
            critic_units=[400, 300],
            policy_lr=1e-3,
            critic_lr=1e-3,
            max_action=1.0,
            noise=1e-2,
            discount=0.99,
            update_rate=1e-2,
            device="/cpu:0",
            name="ddpg"
    ):
        super().__init__(name=name)

        name += "/"
        # policy
        self._policy = Policy(policy_units, dim_state, dim_action, max_action=max_action, name=name+"policy")
        self._policy_target = Policy(policy_units, dim_state, dim_action, max_action=max_action, name=name+"policy_target")
        self._policy_optimizer = tk.optimizers.Adam(learning_rate=policy_lr)
        update.copy(self._policy, self._policy_target)

        # critic
        self._critic = Critic(critic_units, dim_state, dim_action, name=name+"critic")
        self._critic_target = Critic(critic_units, dim_state, dim_action, name=name+"critic_target")
        self._critic_optimizer = tk.optimizers.Adam(learning_rate=critic_lr)
        update.copy(self._critic, self._critic_target)

        # hyper parameters
        self._noise = noise
        self._discount = discount
        self._update_rate = update_rate

        # training condition
        self.device = device

    def get_action(self, state, noise=False):
        assert isinstance(state, np.ndarray)
        assert len(state.shape) == 1

        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self._action_body(state).numpy()
        if noise:
            action += np.random.normal(0.0, self._noise, action.shape).astype(np.float32)
            action = np.clip(action, -self._policy.max_action, self._policy.max_action)
        return action

    @tf.function
    def _action_body(self, state):
        action = self._policy(state)
        action = tf.squeeze(action, axis=0)
        return action

    def train(self, batch):
        state = batch.state
        action = batch.action
        next_state = batch.next_state
        reward = batch.reward
        done = batch.done
        tderror, closs, ploss = self._train_body(state, action, next_state, reward, done)
        return tderror.numpy(), closs.numpy(), ploss.numpy()

    @tf.function
    def _train_body(self, state, action, next_state, reward, done):
        with tf.GradientTape() as tape:
            tderror = self._tderror(state, action, next_state, reward, done)
            critic_loss = tf.reduce_mean(tf.square(tderror))
        critic_grad = tape.gradient(critic_loss, self._critic.trainable_variables)
        self._critic_optimizer.apply_gradients(zip(critic_grad, self._critic.trainable_variables))

        with tf.GradientTape() as tape:
            q = self._critic([state, self._policy(state)])
            policy_loss = - tf.reduce_mean(q)
        policy_grad = tape.gradient(policy_loss, self._policy.trainable_variables)
        self._policy_optimizer.apply_gradients(zip(policy_grad, self._policy.trainable_variables))

        # update.soft_update(
        #     self._critic.trainable_variables,
        #     self._critic_target.trainable_variables,
        #     self._update_rate
        # )
        # update.soft_update(
        #     self._policy.trainable_variables,
        #     self._policy_target.trainable_variables,
        #     self._update_rate
        # )
        update.soft_update(
            self._critic,
            self._critic_target,
            self._update_rate
        )
        update.soft_update(
            self._policy,
            self._policy_target,
            self._update_rate
        )
        return tf.reduce_mean(tderror), critic_loss, policy_loss

    def _tderror(self, state, action, next_state, reward, done):
        not_done = (1 - done)
        target_Q = self._critic_target([next_state, self._policy_target(next_state)])
        target_Q = reward + self._discount * not_done * target_Q
        target_Q = tf.stop_gradient(target_Q)
        current_Q = self._critic([state, action])
        tderror = target_Q - current_Q
        return tderror


def _test_critic():
    units = [32, 32]
    dim_state = 3
    dim_action = 2
    lr = 1e-3

    batch_size = 20
    max_round = 1000

    critic = Critic(units, dim_state, dim_action)
    critic.summary()
    test_state  = np.ones((1, dim_state),  dtype=np.float32)
    test_action = np.ones((1, dim_action), dtype=np.float32)
    Q = critic([test_state, test_action])

    optimizer = tk.optimizers.Adam(learning_rate=lr)

    for round in range(max_round):
        obs = np.random.rand(batch_size, dim_state).astype(np.float32)
        act = np.random.rand(batch_size, dim_action).astype(np.float32)
        ans = np.concatenate([obs, act], axis=1)
        ans = np.mean(ans, axis=1, keepdims=True)
        ans = tf.constant(ans)
        with tf.GradientTape() as tape:
            pre = critic([obs, act])
            loss = tf.reduce_mean(tf.square(pre - ans) / 2)
        grad = tape.gradient(loss, critic.trainable_variables)
        optimizer.apply_gradients(zip(grad, critic.trainable_variables))
        print("{:10.6f}".format(loss.numpy()))


def _test_policy():
    units = [32, 32]
    dim_state = 3
    dim_action = 2
    max_action = 1.0
    lr = 1e-3

    batch_size = 5
    max_round = 1000

    policy = Policy(units, dim_state, dim_action, max_action=max_action)
    policy.summary()

    test_state = np.ones((1, dim_state), dtype=np.float32)
    act = policy(test_state)

    optimizer = tk.optimizers.Adam(learning_rate=lr)

    for round in range(max_round):
        obs = np.random.rand(batch_size, dim_state).astype(np.float32)
        ans = np.mean(obs, axis=1, keepdims=True)
        ans = np.concatenate([ans, -ans], axis=1)
        ans = tf.constant(ans)
        with tf.GradientTape() as tape:
            pre = policy(obs)
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(pre - ans), axis=1) / 2)
        grad = tape.gradient(loss, policy.trainable_variables)
        optimizer.apply_gradients(zip(grad, policy.trainable_variables))
        print("{:10.6f}".format(loss.numpy()))


if __name__ == '__main__':
    # _test_critic()
    _test_policy()

