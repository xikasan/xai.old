# coding: utf-8

import gym
import numpy as np
import xtools as xt
from xai.utils.buffer import ReplayBuffer, BatchLoader


num_step = 200
batch_size = 32


def run():
    env = gym.make("Pendulum-v0")
    xt.info("env", env)

    rb = ReplayBuffer()
    xt.info("ReplayBuffer", rb)

    state = env.reset()

    for i in range(num_step):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        reward = i

        rb(state=state, action=action, next_state=next_state, reward=reward, done=done)
        state = next_state

    loader = BatchLoader(rb, batch_size)
    for batch in loader:
        print(batch.reward)


if __name__ == '__main__':
    run()
