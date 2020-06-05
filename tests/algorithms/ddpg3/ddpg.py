# coding: utf-8

import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from xai.utils import space, update, order, buffer

# =============================================================================
# hyper parameters
#
MAX_EPISODE = int(3e3)
MAX_EP_STEP = 1000
WARMUP = int(1e3)
LR_C = 2e-3
LR_A = 1e-3
GAMMA = 0.99
DECAY = 0.01 # soft update
BF_CAPACITY = int(1e6)
BATCH_SIZE = 100
UNITS_C = [100, 100]
UNITS_A = [100, 100]
# UNITS_C = [32, 32]
# UNITS_A = [32, 32]

RENDER = True
ENV_NAME = "LunarLanderContinuous-v2"
# ENV_NAME = "Pendulum-v0"

TTYPE = tf.float32
NTYPE = np.float32


# =============================================================================
# ddpg
#
# - - - - - - - - - - - - - - - - - - - - - - - -
# critic
class Critic(tk.Model):

    def __init__(self, dim_obs, dim_act, name="critic"):
        super().__init__(name=name)

        self.l1 = tk.layers.Dense(dim_obs*2, activation="relu", name="l1")
        self.l2 = tk.layers.Dense(dim_act*2, activation="relu", name="l2")
        self.l3 = tk.layers.Dense(UNITS_C[0], activation="relu", name="l3")
        self.l4 = tk.layers.Dense(1, activation="linear", name="l4")

        dummy_obs = tk.Input((dim_obs,), dtype=TTYPE)
        dummy_act = tk.Input((dim_act,), dtype=TTYPE)
        self([dummy_obs, dummy_act])

    def call(self, inputs):
        obs, act = inputs
        feature_obs = self.l1(obs)
        feature_act = self.l2(act)
        feature = tf.concat([feature_obs, feature_act], axis=1)
        feature = self.l3(feature)
        feature = self.l4(feature)
        return feature


# - - - - - - - - - - - - - - - - - - - - - - - -
# critic
class Policy(tk.Model):

    def __init__(self, dim_obs, dim_act, max_act, name="policy"):
        super().__init__(name=name)

        self.l1 = tk.layers.Dense(UNITS_A[0], activation="relu", name="l1")
        self.l2 = tk.layers.Dense(UNITS_A[1], activation="relu", name="l2")
        self.l3 = tk.layers.Dense(dim_act,    activation="tanh", name="l3")

        self.max_act = max_act

        dummy_obs = tk.Input((dim_obs,), dtype=TTYPE)
        self(dummy_obs)

    def call(self, inputs):
        feature = self.l1(inputs)
        feature = self.l2(feature)
        feature = self.l3(feature)
        feature = feature * self.max_act
        return feature


# - - - - - - - - - - - - - - - - - - - - - - - -
# agent
class DDPG(tk.Model):

    def __init__(self, dim_obs, dim_act, max_act):
        super().__init__()

        self._policy = Policy(dim_obs, dim_act, max_act, name="policy")
        self._policy_target = Policy(dim_obs, dim_act, max_act, name="policy_target")
        self._policy_optimizer = tk.optimizers.Adam(learning_rate=LR_A)
        update.copy(self._policy, self._policy_target)

        self._critic = Critic(dim_obs, dim_act, name="critic")
        self._critic_target = Critic(dim_obs, dim_act, name="critic_target")
        self._critic_optimizer = tk.optimizers.Adam(learning_rate=LR_C)
        update.copy(self._critic, self._critic_target)

        self.max_act = max_act

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0).astype(NTYPE)
        act = self._get_action_body(obs)
        return act

    def _get_action_body(self, obs):
        act = self._policy(obs)
        act = tf.squeeze(act, axis=0)
        return act

    def train(self, s, a, s_, r, d):
        ploss, closs, tde = self._train_body(s, a, s_, r, d)
        return ploss.numpy(), closs.numpy(), tde.numpy()

    def _train_body(self, s, a, s_, r, d):
        with tf.GradientTape() as policy_tape:
            q = self._critic([s, self._policy(s)])
            policy_loss = - tf.reduce_mean(q)
        policy_grad = policy_tape.gradient(policy_loss, self._policy.trainable_variables)
        self._policy_optimizer.apply_gradients(zip(policy_grad, self._policy.trainable_variables))

        with tf.GradientTape() as critic_tape:
            nd = 1 - d
            target_q = self._critic_target([s_, self._policy_target(s_)])
            target_q = r + GAMMA * nd * target_q
            target_q = tf.stop_gradient(target_q)
            current_q = self._critic([s, a])
            tde = target_q - current_q
            critic_loss = tf.reduce_mean(tf.square(tde))
        critic_grad = critic_tape.gradient(critic_loss, self._critic.trainable_variables)
        self._critic_optimizer.apply_gradients(zip(critic_grad, self._critic.trainable_variables))

        update.soft_update(self._critic, self._critic_target, DECAY)
        update.soft_update(self._policy, self._policy_target, DECAY)

        return policy_loss, critic_loss, tf.reduce_mean(tde)


# make python silent
order.please()

# env
env = gym.make(ENV_NAME)
dim_obs = space.get_size(env.observation_space)
dim_act = space.get_size(env.action_space)
max_act = env.action_space.high

# agent
agent = DDPG(dim_obs, dim_act, max_act)
var = 3  # control exploration

# replay buffer
rb = buffer.ReplayBuffer(BF_CAPACITY, enable_extend=False, dtype=NTYPE)

for ep in range(MAX_EPISODE):
    obs = env.reset()
    ep_reward = 0

    for es in range(MAX_EP_STEP):
        env.render()
        act = agent.get_action(obs)
        act = np.random.normal(act, var)
        act = np.clip(act, -max_act, max_act)
        obs_, reward, done, _ = env.step(act)
        ep_reward += reward

        rb.store(s=obs, a=act, s_=obs_, r=reward, d=done).flush()
        obs = obs_

        if len(rb) >= WARMUP:
            var *= 0.9995
            batch = rb.sample(BATCH_SIZE)
            agent.train(batch.s, batch.a, batch.s_, batch.r, batch.d)

        if done:
            break

    print("episode:{:4.0f} reward:{:10.4f} var:{:6.4f}".format(ep, ep_reward, var))
