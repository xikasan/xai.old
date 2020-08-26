# coding: utf-8

import numpy as np


class Agent:

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.model = None
        self.initialized = None

    def initialize(self, experiences):
        raise NotImplementedError()

    def estimate(self, state):
        raise NotImplementedError()

    def update(self, experiences, gamma):
        raise NotImplementedError()

    def policy(self, state):
        raise NotImplementedError()

    def play(self, env, episode_count=5, render=True):
        for episode in range(episode_count):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if render:
                    env.render()
                action = self.policy(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                state = next_state
            else:
                print("[Episode] Episode: {:4.0f}\t Reward: {:10.5f}".format(episode, episode_reward))
