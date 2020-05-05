# coding: utf-8

import gym


def get_size(space):
    if isinstance(space, gym.spaces.Box):
        return space.high.shape[0]
    if isinstance(space, gym.spaces.Discrete):
        return space.n
    raise ValueError("Unknown space type {} is given.".format(type(space)))
