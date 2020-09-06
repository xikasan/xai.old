# coding: utf-8

import numpy as np
import xtools as xt
from xai.policy.categorical import CategoricalPolicy


def run():
    policy = CategoricalPolicy([32, 32], 3)
    xt.info(policy)

    state = np.random.rand(3).astype(np.float32)
    xt.info("prob", policy(state.reshape((1, -1))))
    for _ in range(10):
        action, prob = policy.get_noisy_action(state)
        xt.info("action", action)


if __name__ == '__main__':
    run()
