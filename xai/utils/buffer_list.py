# coding: utf-8

import xsim
import numpy as np


class ReplayBuffer:

    BASIC_KEYS = ["state", "action", "next_state", "reward", "done"]

    def __init__(self, keys=None, dtype=np.float32):
        keys = self.BASIC_KEYS if keys is None else keys
        assert isinstance(keys, list)
        self._default_keys = keys
        self._buf = None
        self.dtype = dtype
        self.reset()

    def __len__(self):
        key = list(self._buf.keys())[0]
        return len(self._buf[key])

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            assert len(args) == len(self._buf.keys()), \
                "Number of input data {} and stored key {} is not much".format(
                    len(args), len(self._buf.keys())
                )
            for buf, val in zip(self._buf.values(), args):
                buf.append(val)
            return

        if len(kwargs) > 0:
            assert len(kwargs) == len(self._buf.keys()), \
                "Number of input data {} and stored key {} is not much".format(
                    len(kwargs), len(self._buf.keys())
                )
            for key, val in kwargs.items():
                self._buf[key].append(val)
            return

        raise ValueError("No data had been input")

    def reset(self):
        self._buf = {key: [] for key in self._default_keys}

    def buffer(self):
        return {key: np.asarray(val, dtype=self.dtype) for key, val in self._buf.items()}

    def get(self, key):
        return np.asarray(self._buf[key], dtype=self.dtype)

    def append_data(self, key, value):
        self._buf[key].append(value)

    def append_column(self, key, values, ignore_length=False):
        if not ignore_length:
            assert len(values) == len(self), \
                "Data length {} is not much to buffer length {}".format(len(values), len(self))
        self._buf[key] = values


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
