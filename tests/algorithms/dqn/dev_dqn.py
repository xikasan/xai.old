# coding: utf-8

import numpy as np
import xtools as xt
import gym
import tensorflow as tf
from xai.algorithms.dqn import DQN, QNet
from xai.utils import space, buffer


def train(conf):
    cf = xt.Config(conf)

    env = gym.make(cf.env.name)
    test_env = gym.make(cf.env.name)
    dim_obs = space.get_size(env.observation_space)
    dim_act = space.get_size(env.action_space)

    agent = DQN(cf.dqn.units, dim_act, dim_obs, gamma=cf.dqn.gamma, epsilon=cf.dqn.epsilon)

    # saver
    if cf.save.do:
        cf.save.path = xt.make_dirs_current_time(cf.save.path)
        writer = tf.summary.create_file_writer(cf.save.path)
        writer.set_as_default()
        checkpoint = tf.train.Checkpoint(model=agent.model())
        manager = tf.train.CheckpointManager(
            checkpoint,
            xt.join(cf.save.path, "model"),
            max_to_keep=cf.save.model.num
        )

    rb = buffer.ReplayBuffer()
    step = tf.Variable(0, dtype=tf.int64)

    while True:
        obs = env.reset()
        episode_step = 0
        episode_reward = 0

        while True:
            act = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(act)
            episode_reward += reward
            episode_step += 1

            rb.store(state=obs, action=act, next_state=next_obs, reward=reward, done=done).flush()

            # train
            if step >= cf.train.step.warmup:
                batch = rb.sample(cf.train.batch)
                loss  = agent.train(batch)
                tf.summary.scalar("train/loss", tf.constant(loss), step=step)

            # test
            if (step.numpy() % cf.test.interval) == 0:
                test_step, test_reward = test(cf, agent, test_env)
                tf.summary.scalar("test/episode_step", tf.constant(test_step),   step=step)
                tf.summary.scalar("test/reward",       tf.constant(test_reward), step=step)
                print("test: step", test_step, "reward", test_reward)

            step = step + 1
            obs = next_obs

            if step > cf.train.step.max:
                return

            if done:
                env.close()
                break

        tf.summary.scalar("train/episode_step",   tf.constant(episode_step),   step=step)
        tf.summary.scalar("train/episode_reward", tf.constant(episode_reward), step=step)
        print("step:", step, "episode[ step:", episode_step, "reward:", episode_reward, "]")


def test(cf, agent, env):
    obs = env.reset()

    test_step = 0
    test_reward = 0
    while True:
        act = agent.select_action_greedy(obs)
        next_obs, reward, done, _ = env.step(act)

        test_step += 1
        test_reward += reward

        if cf.test.render:
            env.render()

        if done:
            break

        obs = next_obs

    env.close()
    return test_reward, test_step


if __name__ == '__main__':
    xt.go_to_root()
    config = "tests/algorithms/dqn/config_dev.yaml"
    train(config)
