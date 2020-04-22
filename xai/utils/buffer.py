# coding: utf-8

import numpy as np
import xsim


def ReplayBuffer(size=1000, enable_extend=False, dtype=np.float32):
    if enable_extend:
        return ExtendableReplayBuffer(size, dtype)
    raise ValueError("enable_extend=False is not ready")


class ExtendableReplayBuffer(xsim.Logger):

    def __init__(self, size=1000, dtype=np.float32):
        super().__init__(size, dtype)

    def sample(self, size):
        idx = np.random.randint(0, self._counter, size)
        return xsim.Batch.make({key: val[idx, :] for key, val in self._buf.items()})
