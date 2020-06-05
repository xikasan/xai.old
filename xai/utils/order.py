# coding: utf-8

import os
import gym


def please():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gym.logger.set_level(gym.logger.ERROR)
