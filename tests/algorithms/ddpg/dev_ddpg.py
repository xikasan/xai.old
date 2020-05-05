# coding: utf-8

import gym
import xtools as xt
import tensorflow as tf
from xai.algorithms.ddpg import *
from xai.utils import space, buffer


def run(conf):
    cf = xt.Config(conf)
    xt.info("config", cf)

    # -----------------------------------------------------
    # prepare result directory and TensorBoard
    if cf.save.do:
        save_dir = xt.make_dirs_current_time(xt.join(cf.save.directory, cf.env.name))
        writer = tf.summary.create_file_writer(save_dir)
        writer.set_as_default()

    # -----------------------------------------------------
    # preparation
    env = gym.make(cf.env.name)
    test_env = gym.make(cf.env.name)
    xt.info("env", env)

    agent = generate_ddpg(cf, env)
    xt.info("ddpg", agent)

    rb = buffer.ReplayBuffer()
    xt.info("ReplayBuffer", rb)

    # -----------------------------------------------------
    # main loop
    main(cf, agent, rb, env, test_env)

    # -----------------------------------------------------
    # dispose
    env.close()
    test_env.close()


def main(cf, agent, rb, env, test_env):
    obs = env.reset()

    episode_num = 0
    episode_step = 0
    episode_reward = 0

    for s in range(cf.train.step.max):
        step = s + 1
        obs, reward, done = env_step(agent, env, rb, obs)
        episode_reward += reward
        episode_step += 1

        if done or episode_step >= cf.env.step.max:
            print(
                "[train] step:{:6.0f}".format(step),
                "reward:{:8.2f}".format(episode_reward),
                "step:{:4.0f}".format(episode_step)
            )
            # write episode result
            tf.summary.scalar("train/reward", episode_reward, step=step)
            tf.summary.scalar("train/step",   episode_step,   step=step)
            # reset env
            obs = env.reset()
            episode_num += 1
            episode_step = 0
            episode_reward = 0

        # check warm up
        if step < cf.train.step.start:
            continue

        critic_loss, policy_loss = train_once(cf, agent, rb)

        if (step % cf.save.interval) == 0:
            tf.summary.scalar("train/critic_loss", critic_loss, step=step)
            tf.summary.scalar("train/policy_loss", policy_loss, step=step)

        if (step % cf.test.interval) == 0:
            test_reward, test_step = test_once(cf, agent, test_env)
            tf.summary.scalar("test/reward", tf.constant(test_reward), step=step)
            tf.summary.scalar("test/step", tf.constant(test_step), step=step)


def train_once(cf, agent, rb):
    batch = rb.sample(cf.train.batch.size)
    return agent.train(batch)


def env_step(agent, env, rb, obs):
    act = agent.select_noisy_action(obs)
    next_obs, reward, done, _ = env.step(act)
    rb.store(state=obs, action=act, next_state=next_obs, reward=reward, done=done).flush()
    return next_obs, reward, done


def test_once(cf, agent, test_env):
    test_reward = 0
    obs = test_env.reset()
    for test_step in range(1, cf.env.step.max):
        act = agent.select_action(obs)
        next_obs, reward, done, _ = test_env.step(act)
        test_reward += reward

        if cf.test.render:
            test_env.render()

        if done:
            break

        obs = next_obs

    return test_reward, test_step


def generate_ddpg(cf, env):
    dcf = cf.ddpg
    return DDPG(
        dcf.critic.units,
        dcf.policy.units,
        space.get_size(env.action_space),
        space.get_size(env.observation_space),
        env.action_space.high,
        critic_lr=dcf.critic.lr,
        policy_lr=dcf.policy.lr,
        update_rate=dcf.update_rate,
        discount=dcf.discount,
        noise=dcf.noise
    )


if __name__ == '__main__':
    xt.go_to_root()
    config = "tests/algorithms/ddpg/config.yaml"
    run(config)
