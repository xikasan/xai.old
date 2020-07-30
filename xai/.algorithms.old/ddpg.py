# coding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from ..utils import update
# from xai.utils import update


class Critic(tk.Model):

    def __init__(self, units, dim_action, dim_state, name="Critic"):
        super().__init__(name=name)

        name += "/"
        self.l1 = tk.layers.Dense(units[0], activation="relu",   name=name+"L1")
        self.l2 = tk.layers.Dense(units[1], activation="relu",   name=name+"L2")
        self.lo = tk.layers.Dense(1,        activation="linear", name=name+"Lo")

        dummy_state  = tk.Input((dim_state, ), dtype=tf.float32)
        dummy_action = tk.Input((dim_action,), dtype=tf.float32)
        self([dummy_state, dummy_action])

    def call(self, inputs):
        feature = tf.concat(inputs, axis=1)
        feature = self.l1(feature)
        feature = self.l2(feature)
        feature = self.lo(feature)
        return feature


class Policy(tk.Model):

    def __init__(self, units, dim_action, dim_state, max_action=1.0, name="Policy"):
        super().__init__(name=name)

        name += "/"
        self.l1 = tk.layers.Dense(units[0],   activation="relu", name=name+"L1")
        self.l2 = tk.layers.Dense(units[1],   activation="relu", name=name+"L2")
        self.lo = tk.layers.Dense(dim_action, activation="tanh", name=name+"Lo")

        self.max_action = max_action

        dummy_state = tk.Input((dim_state, ), dtype=tf.float32)
        self(dummy_state)

    def call(self, inputs):
        feature = self.l1(inputs)
        feature = self.l2(feature)
        feature = self.lo(feature)
        feature = feature * self.max_action
        return feature


class DDPG(tk.Model):

    def __init__(
            self,
            critic_units,
            policy_units,
            dim_action,
            dim_state,
            max_action=1.0,
            critic_lr=1e-3,
            policy_lr=1e-3,
            update_rate=0.01,
            discount=0.99,
            noise=0.01,
            name="DDPG"
    ):
        super().__init__(name=name)

        name += "/"
        self._critic        = Critic(critic_units, dim_action, dim_state, name=name+"Critic")
        self._critic_target = Critic(critic_units, dim_action, dim_state, name=name+"CriticTarget")
        update.copy(self._critic, self._critic_target)
        self._policy        = Policy(policy_units, dim_action, dim_state, max_action=max_action, name=name+"Policy")
        self._policy_target = Policy(policy_units, dim_action, dim_state, max_action=max_action, name=name+"PolicyTarget")
        update.copy(self._policy, self._policy_target)

        self._critic_lr = tf.Variable(critic_lr, dtype=tf.float32)
        self._policy_lr = tf.Variable(policy_lr, dtype=tf.float32)
        self._critic_optimizer = tk.optimizers.Adam(learning_rate=self._critic_lr)
        self._policy_optimizer = tk.optimizers.Adam(learning_rate=self._policy_lr)
        # self._critic_optimizer = tf.optimizers.Adam(learning_rate=critic_lr)
        # self._policy_optimizer = tf.optimizers.Adam(learning_rate=policy_lr)

        self.update_rate = update_rate
        self.discount = discount
        self.noise = noise

    def select_action(self, state):
        assert isinstance(state, np.ndarray)
        assert len(state.shape) == 1

        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self._select_action_body(state).numpy()
        return action

    def select_noisy_action(self, state):
        action = self.select_action(state)
        action = action + np.random.normal(0, self.noise, size=action.shape)
        action = np.clip(action, -self._policy.max_action, self._policy.max_action)
        return action

    @tf.function
    def _select_action_body(self, state):
        action = self._policy(state)
        action = tf.squeeze(action, axis=0)
        return action

    def train(self, batch):
        state = batch.state
        action = batch.action
        next_state = batch.next_state
        reward = batch.reward
        done = batch.done.astype(np.float32)
        return self._train_body(state, action, next_state, reward, done)

    @tf.function
    def _train_body(self, state, action, next_state, reward, done):
        # critic
        with tf.GradientTape() as critic_tape:
            tderror = self.tderror(state, action, next_state, reward, done)
            critic_loss = tf.reduce_mean(tf.square(tderror))
        critic_grad = critic_tape.gradient(critic_loss, self._critic.trainable_variables)
        self._critic_optimizer.apply_gradients(zip(critic_grad, self._critic.trainable_variables))

        # policy
        with tf.GradientTape() as policy_tape:
            policy_loss = self._critic([state, self._policy(state)])
            policy_loss = - tf.reduce_mean(policy_loss)
        policy_grad = policy_tape.gradient(policy_loss, self._policy.trainable_variables)
        self._policy_optimizer.apply_gradients(zip(policy_grad, self._policy.trainable_variables))

        update.soft_update(self._critic, self._critic_target, self.update_rate)
        update.soft_update(self._policy, self._policy_target, self.update_rate)

        return tf.reduce_mean(tderror), critic_loss, policy_loss

    def tderror(self, state, action, next_state, reward, done):
        target_q = self.discount * (1 - done) * self._critic_target([
            next_state,
            self._policy_target(next_state)
        ])
        target_q = reward + target_q
        current_q = self._critic([state, action])
        return tf.stop_gradient(target_q) - current_q


if __name__ == '__main__':
    import xtools as xt
    xt.go_to_root()
    critic = Critic([8, 8], 3, 2)
    critic.summary()

    opt = tf.optimizers.Adam(learning_rate=1e-3)

    save_dir = xt.make_dirs_current_time("tests/algorithms/ddpg/result/test")
    writer = tf.summary.create_file_writer(save_dir)
    writer.set_as_default()

    for step in range(10000):
        act = np.random.rand(1, 2).astype(np.float32)
        obs = np.random.rand(1, 3).astype(np.float32)
        rwd = np.concatenate([obs, act], axis=1)
        rwd = np.sum(np.square(obs)) + np.sum(np.square(act))

        with tf.GradientTape() as tape:
            val = critic([obs, act])
            error = val - rwd
            loss  = tf.square(error) / 2
            loss  = tf.reduce_mean(loss)
        grad = tape.gradient(loss, critic.trainable_variables)
        opt.apply_gradients(zip(grad, critic.trainable_variables))

        tf.summary.scalar("train/loss", loss, step=step)

        if ((step+1)%100) == 0:
            print("step:{:5} \t loss:{:10.6f}".format(step+1, loss.numpy()))
        # exit()
