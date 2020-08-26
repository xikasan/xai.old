# coding: utf-8

import xsim
import numpy as np


class ReplayBuffer:
    BASIC_KEYS = ["state", "action", "next_state", "reward", "done"]

    def __init__(self, keys=None, max_size=1024, dtype=np.float32):
        keys = self.BASIC_KEYS if keys is None else keys
        assert isinstance(keys, list)
        self.max_size = max_size
        self._default_keys = keys
        self._buf = None
        self._counter = None
        self._is_full = False
        self.dtype = dtype
        self.reset()

    def __len__(self):
        if self._is_full:
            return self.max_size
        return self._counter

    def __call__(self, **kwargs):
        # store data
        for key, val in kwargs.items():
            val = np.squeeze(val)
            self._add_key_if_not_exist(key, val)
            self._buf[key][self._counter, :] = val

        # count up
        self._counter += 1
        if (self._counter % self.max_size) == 0:
            self._counter = 0
            self._is_full = True
        return

    def reset(self):
        self._buf = {}
        self._counter = 0
        self._is_full = False

    def buffer(self):
        return {key: val[0:self._counter] for key, val in self._buf.items()}

    def get(self, key):
        return self._buf[key].copy()

    def append_column(self, key, values):
        self._add_key_if_not_exist(key, values[0])
        temp_counter = self._counter
        for val in reversed(values):
            # count down
            temp_counter -= 1
            if temp_counter < 0:
                temp_counter += self.max_size

            self._buf[key][temp_counter, :] = np.squeeze(val)

    def _add_key_if_not_exist(self, key, val):
        # existing check
        if key in self._buf.keys():
            return

        # add new buffer
        shape = val.shape
        shape = 1 if len(shape) == 0 else shape[0]

        # add new buffer
        self._buf[key] = np.zeros((self.max_size, shape), dtype=self.dtype)


class BatchLoader:

    def __init__(self, buffer, batch_size, dtype=np.float32):
        self.batch_size = batch_size
        self._buf = buffer
        self._counter = None
        self._data = None
        self.dtype = dtype

    def __len__(self):
        return int(np.ceil(len(self._buf) / self.batch_size))

    def __iter__(self):
        self._data = self._buf.buffer()
        self._data_size = len(self._buf)
        self._counter = 0
        self._indices = np.random.permutation(self._data_size)
        return self

    def __next__(self):
        if self._counter >= self._data_size:
            raise StopIteration()

        batch = self._retrieve()
        self._counter += self.batch_size
        return batch

    def _retrieve(self):
        idxs = self._indices[self._counter:self._counter+self.batch_size]
        batch = {key: np.asarray(val[idxs]) for key, val in self._data.items()}
        return xsim.Batch.make(batch)
