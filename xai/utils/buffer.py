# coding: utf-8

import numpy as np
import xsim


def ReplayBuffer(size=1000, enable_extend=False, dtype=np.float32):
    if enable_extend:
        return ExtendableReplayBuffer(size, dtype)
    return FixedLengthReplayBuffer(size, dtype)


class ExtendableReplayBuffer(xsim.Logger):

    def __init__(self, size=1000, dtype=np.float32):
        super().__init__(size, dtype)

    def sample(self, size):
        idx = np.random.randint(0, self._counter, size)
        return xsim.Batch.make({key: val[idx, :] for key, val in self._buf.items()})


class FixedLengthReplayBuffer:

    def __init__(self, capacity=1000, dtype=np.float32):
        self.dtype = dtype
        self.capacity = capacity
        self._buf = {}

        self._pointer = 0
        self._is_full = False
        self._current_data = {}

    def __len__(self):
        return self.capacity if self._is_full else self._pointer

    def store(self, **kwargs):
        for key, val in kwargs.items():
            self._current_data[key] = np.squeeze(val)
        return self

    def flush(self):
        for key, val in self._current_data.items():
            self._add_key_if_not_exist(key, val)
            self._buf[key][self._pointer, :] = val
        self._pointer = (self._pointer + 1) % self.capacity
        if self._pointer == 0:
            self._is_full = True
        self._current_data = {}

    def sample(self, size):
        idx = self.capacity if self._is_full else self._pointer
        idx = np.random.choice(np.arange(idx), size, replace=False)
        return xsim.Batch.make({key: val[idx, :] for key, val in self._buf.items()})

    def _add_key_if_not_exist(self, key, val):
        # existing check
        if key in self._buf.keys():
            return

        # find width
        shape = val.shape
        shape = 1 if len(shape) == 0 else shape[0]

        # add new buffer
        self._buf[key] = np.zeros((self.capacity, shape), dtype=self.dtype)
