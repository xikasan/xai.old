# coding: utf-8

import numpy as np
import tensorflow as tf
from collections import deque
from .experience import Experience


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
            self.begin_episode(i, agent)
            while not done:
                if render:
                    env.render()

                if self.training and observe_interval > 0 and \
                        (self.training_count == 1 or self.training_count % observe_interval == 0):
                    frames.append(state)

                action = self.policy(state)
                next_state, reward, done, _ = env.step(action)
                experience = Experience(state, action, next_state, reward, done)
                self.experiences.append(experience)

                if not self.training and len(self.experiences) == self.buffer_size:
                    self.begin_train(i, agent)
                    self.training = True

                self.step(i, step_count, agent, experience)

                state = next_state
                step_count += 1
            else:
                self.end_episode(i, step_count, agent)

                if not self.training and initial_count > 0 and i >= initial_count:
                    self.begin_train(i, agent)
                    self.training = True

                if self.training:
                    if len(frames) > 0:
                        self.logger.write_image(self.training_count, frames)
                        frames = []

    def begin_episode(self, episode, agent):
        pass

    def begin_train(self, episode, agent):
        pass

    def step(self, episode, step_count, agent, experience):
        pass

    def end_episode(self, episode, step_count, agent):
        pass

    def is_event(self, count, interval):
        return count != 0 and (count % interval) == 0

    def get_recent(self, count):
        recent = range(len(self.experiences) - count, len(self.experiences))
        return [self.experiences[i] for i in recent]
