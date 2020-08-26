# coding: utf-8

import xsim
import numpy as np


class ReplayBuffer:

    BASIC_KEYS = [
        "state",
        "action",
        "next_state",
        "reward",
        "done"
    ]

    def __init__(self, keys):
        assert isinstance(keys, list)
        self._buf = {key: [] for key in keys}

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            assert len(args) == len(self._buf.keys()), \
                "Number of input data {} and memory key {} is not much".format(len(args), len(self._buf.keys()))
            for buf, val in zip(self._buf.values(), args):
                buf.append(val)
            return

        if len(kwargs) > 0:
            assert len(kwargs) == len(self._buf.keys()), \
                "Number of input data {} and memory key {} is not much".format(len(kwargs), len(self._buf.keys()))
            for key, val in kwargs.items():
                self._buf[key].append(val)
            return

        raise ValueError("No data had been input")

    def reset(self):
        self._buf = {key: [] for key in self._buf.keys()}
        return self

    def buffer(self):
        return self._buf

    def get(self, key):
        return self._buf[key].copy()

    def append_date(self, key, value):
        self._buf[key].append(value)

    def set_data(self, key, values):
        self._buf[key] = values


class BatchLoader:

    def __init__(self, buffer, batch_size, dtype=np.float32):
        self.batch_size = batch_size
        self._buf = buffer
        self._indices = None
        self._counter = None
        self._data = None
        self.dtype = dtype
        self._data_size = None

    def __len__(self):
        data_size = self._data_size
        if data_size is None:
            data = self._buf if not isinstance(self._buf, ReplayBuffer) else self._buf.buffer()
            data_size = len(data[list(data.keys())[0]])
        return int(np.ceil(data_size / self.batch_size))

    def __iter__(self):
        self._counter = 0
        self._indices = np.arange(1)
        return self

    def __next__(self):
        if self._counter >= self._data_size:
            raise StopIteration

        batch = self._retrieve()
        self._counter += self.batch_size
        return batch

    def _retrieve(self):
        batch = {
            key: np.asarray(val[self._counter:self._counter+self.batch_size]).astype(self.dtype)
            for key, val in self._data.items()
        }
        return xsim.Batch.make(batch)
