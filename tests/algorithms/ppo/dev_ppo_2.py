# coding: utf-8

import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk

import xsim
import xtools as xt
from xai.utils.distribution import GaussianDistribution
from xai.utils.buffer import ReplayBuffer


ENV_NAME = "MountainCarContinuous-v0"

NUM_EPOCH = 1000
EPISODE_STEP = 100
BATCH_SIZE = 32
BATCH_LABELS = [
    "observation", "action", "next_observation",
    "reward", "done", "value", "log_pi", "entropy",
    "advantage"
]


class Model(tk.Model):

    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, act_dim):
        super().__init__()

        self.l1 = tk.layers.Dense(32, activation="relu", name="L1")
        self.l2 = tk.layers.Dense(32, activation="relu", name="L2")
        self.lo_critic = tk.layers.Dense(1, name="Lo_critic")
        self.lo_policy_mean = tk.layers.Dense(act_dim, name="Lo_policy_mean")
        self.lo_policy_log_std = tk.layers.Dense(act_dim, name="Lo_policy_log_std")

    def call(self, states, actions=None):
        feature = self.l1(states)
        feature = self.l2(feature)

        values = self.lo_critic(feature)
        params = self._compute_dist(feature)

        if actions is None:
            actions = GaussianDistribution.sample(params)

        log_pis = GaussianDistribution.log_likelihood(actions, params)
        entropy = GaussianDistribution.entropy(params)

        return actions, log_pis, entropy, values

    def _compute_dist(self, feature):
        mean = self.lo_policy_mean(feature)
        log_std = self.lo_policy_log_std(feature)
        log_std = tf.clip_by_value(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return {"mean": mean, "log_std": log_std}


class PPO:

    def __init__(
            self,
            obs_dim,
            act_dim,
            gamma,
            gae_lambda=1,
            learning_rate=1e-3,
            clip_param=0.2
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = learning_rate
        self.clip_param = clip_param

        self.model = Model(act_dim)

    def get_action_and_value(self, observation):
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)
        observation = observation.astype(np.float32)

        return self.model(observation)

    def compute_advantage(self, rewards, values, dones):
        masks = 1 - dones
        deltas = rewards + self.gamma * masks * values[1:, :] - values[:-1, :]
        gae = 0
        advantages = np.zeros_like(deltas)
        for i in reversed(range(len(rewards))):
            gae = deltas[i, 0] + self.gamma * self.gae_lambda * gae
            advantages[i, 0] = gae
        return advantages


def run():
    env = gym.make(ENV_NAME)

    agent = PPO(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        1.0
    )
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)

    local_bf = xsim.Logger()

    def env_step(action):
        next_obs, reward, done, _ = env.step(action)
        return next_obs, reward, done

    def finish_episode(last_value):
        bf = local_bf.latest(len(local_bf))
        values = np.append(bf.value, last_value, axis=0)
        advantages = agent.compute_advantage(bf.reward, values, bf.done)
        local_bf._add_key_if_not_exist("advantage", advantages[0, :])
        local_bf._buf["advantage"][:len(advantages), :] = advantages
        return local_bf

    for epoch in range(NUM_EPOCH):

        observation = env.reset()
        for _ in range(EPISODE_STEP):
            local_bf.store(observation=observation)
            action, log_pi, entropy, value = agent.get_action_and_value(observation)
            action = action.numpy().reshape(1)

            observation, reward, done = env_step(action)

            local_bf.store(
                action=action,
                next_observation=observation,
                reward=reward,
                done=done,
                value=value,
                log_pi=log_pi,
                entropy=entropy
            ).flush()

            if done:
                break

        _, _, _, last_value = agent.get_action_and_value(observation)
        last_value = last_value.numpy()
        train_bf = finish_episode(last_value)

        indices = np.arange(len(train_bf))
        np.random.shuffle(indices)

        train_bf = train_bf.buffer()

        def make_batch(idx):
            law_batch = {
                label: train_bf[label][idx, :] for label in BATCH_LABELS
            }
            return xsim.Batch.make(law_batch)

        i = 0
        while True:
            idxs = indices[i:i+BATCH_SIZE]
            i += BATCH_SIZE

            # training
            batch = make_batch(idxs)
            with tf.GradientTape() as tape:
                _, log_pi_new, entropy, values = agent.get_action_and_value(batch.observation)
                log_pi_new = tf.expand_dims(log_pi_new, axis=-1)
                ratio = tf.exp(log_pi_new - batch.log_pi)

                surr1 = - ratio * batch.advantage
                suur2 = - tf.clip_by_value(
                    ratio, 1 - agent.clip_param, 1 + agent.clip_param
                ) * batch.advantage

                policy_loss = tf.reduce_mean(tf.maximum(surr1, suur2))  # - 0.1 * tf.reduce_mean(entropy)
                critic_loss = tf.reduce_mean(0.5 * tf.square(values - batch.advantage))

                loss = policy_loss + critic_loss
                # print(loss)

            gradient = tape.gradient(loss, agent.model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, agent.model.trainable_variables))

            if i > len(local_bf):
                break

        print("Epoch: {:4.0f}".format(epoch), "Reward:", np.sum(local_bf.buffer()["reward"]))
        local_bf = xsim.Logger()


if __name__ == '__main__':
    run()
