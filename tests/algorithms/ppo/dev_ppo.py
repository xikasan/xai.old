# coding: utf-8

import xsim
import numpy as np
import xtools as xt
import tensorflow as tf
tkl = tf.keras.layers

import gym
import aerospace_gym

from xai.policy.gaussian_actor import GaussianPolicy


ENV_NAME = "MountainCarContinuous-v0"
ENV_DUE = 10

DECAY_FACTOR = 0.9


class Critic(tf.keras.Model):

    def __init__(self):
        super().__init__(name="critic")
        self.l1 = tkl.Dense(32, activation="relu",   name="L1")
        self.l2 = tkl.Dense(32, activation="relu",   name="L2")
        self.lo = tkl.Dense(1,  activation="linear", name="Lo")

    def call(self, inputs):
        feature = self.l1(inputs)
        feature = self.l2(feature)
        value = self.lo(feature)
        return tf.squeeze(value, axis=1)


def run():
    xt.info("developping ppo")

    env = gym.make(ENV_NAME)
    xt.info("env", env)
    obs = env.reset()

    critic = Critic()
    policy = GaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0])

    obs = obs.astype(np.float32)
    inputs = obs.reshape(1, -1)
    values = critic(inputs)
    act, log_ps, _ = policy(inputs)

    local_bf = xsim.Logger()

    while True:
        state = obs.reshape(1, -1)
        act, log_pi, params = policy(state)
        obs_, reward, done, _ = env.step(act)

        local_bf.store(obs=obs, act=act, next_obs=obs_, reward=reward, done=done).flush()

        if done:
            train_bf = finish_horizon(local_bf)
            # for batch in train_bf:
            #     pass
            break

        obs = obs_

    xt.info("the end")


def finish_horizon(bf):
    xt.info("finishing horizon")
    bf = bf.latest(len(bf))
    returns = [discounted_sum(bf.reward[tgt:]) for tgt in range(bf.size)]
    returns = np.reshape(returns, (bf.size, 1))
    bf.append("returns", returns)
    return bf


def discounted_sum(rewards):
    decay_factors = DECAY_FACTOR ** np.arange(len(rewards)).reshape((len(rewards), 1))
    return np.sum(rewards * decay_factors).astype(np.float32)


if __name__ == '__main__':
    run()
