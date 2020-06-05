# coding: utf-8

import gym
import numpy as np


def run():
    env = gym.make("LunarLanderContinuous-v2")
    env.reset()

    for time in range(1000):
        act = env.action_space.sample()
        env.step(act)
        env.render()
    env.close()


if __name__ == '__main__':
    run()
