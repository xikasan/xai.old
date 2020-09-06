# coding: utf-8

import gym
import xsim
import xtools as xt
import numpy as np
import tensorflow as tf
from xai.algorithms.ddpg2 import DDPG
from xai.utils import order, space, buffer

order.please()


def run():
    conf = "tests/algorithms/ddpg2/config_bipedal.yaml"
    cf = xt.Config(conf)

    env = gym.make(cf.env.name)
    test_env = gym.make(cf.env.name)
    xt.info(env)

    dim_state = space.get_size(env.observation_space)
    dim_action = space.get_size(env.action_space)

    dcf = cf.ddpg
    agent = DDPG(
        dim_state,
        dim_action,
        policy_units=dcf.policy.units,
        critic_units=dcf.critic.units,
        policy_lr=dcf.policy.lr,
        critic_lr=dcf.critic.lr,
        max_action=dcf.max_action,
        noise=dcf.noise,
        discount=dcf.discount,
        update_rate=dcf.update_rate
    )

    bf = buffer.ReplayBuffer()

    obs = env.reset()
    step = 0

    episode_tderror = 0
    episode_step = 0
    episode_reward = 0
    episode = 1

    while True:
        act = agent.get_action(obs, noise=True)
        if len(bf) <= cf.train.step.warmup:
            act = env.action_space.sample()
        # print(act)
        next_obs, reward, done, _ = env.step(act)
        # reward = next_obs[0]
        env.render()
        # print(reward)
        bf.store(state=obs, action=act, next_state=next_obs, reward=reward, done=done).flush()
        episode_step += 1
        episode_reward += reward

        obs = next_obs

        if done or episode_step == cf.env.step:
            obs = env.reset()
            print("step:{:8.0f} episode:{:4.0f} reward:{:12.3f} TDErrpr:{:12.6f}".format(
                step, episode, episode_reward, episode_tderror/episode_step))
            episode_tderror = 0
            episode_reward = 0
            episode_step = 0
            episode += 1

        if len(bf) <= cf.train.step.warmup:
            continue

        step += 1

        batch = bf.sample(cf.train.batch_size)
        tde, closs, ploss = agent.train(batch)
        episode_tderror += tde

        if (step % cf.test.interval) == 0:
            test_obs = test_env.reset()
            test_episode_reward = 0
            for test_step in range(cf.env.step):
                test_act = agent.get_action(test_obs)
                test_next_obs, test_reward, test_done, _ = test_env.step(test_act)
                test_episode_reward += test_reward
                if cf.test.render:
                    test_env.render()
                if test_done:
                    break
                test_obs = test_next_obs
            print("[test] reward:{:12.3f}".format(test_episode_reward))

        if step == cf.train.step.max:
            test_env.close()
            break


if __name__ == '__main__':
    xt.go_to_root()
    run()
