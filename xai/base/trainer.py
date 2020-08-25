# coding: utf-8

import numpy as np
import tensorflow as tf
from collections import deque


class Trainer:

    def __init__(self, buffer_size=1024, batch_size=32, gamma=0.9, report_interval=10, log_dir=""):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.report_interval = report_interval
        self.logger = None
        self.experiences = deque(maxlen=buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []

    def train_loop(self, env, agent, episode=200, initial_count=-1, render=False, observe_interval=0):
        self.experiences = deque(maxlen=self.buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []
        frames = []

        for i in range(episode):
            state = env.reset()
            done = False
            step_count = 0
            self.episode_begin(i, agent)
            while not done:
                if render:
                    env.render()

                if self.training and observe_interval > 0 and \
                    (self.training_count == 1 or self.training_count % observe_interval == 0):
                    frames.append(state)

                action = self.policy(state)
                next_state, reward, done, _ = env.step(action)
                experience = None

    def episode_begin(self, episode, agent):
        pass
