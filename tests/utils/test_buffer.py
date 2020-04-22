# coding: utf-8

import numpy as np
from xai.utils.buffer import ReplayBuffer


def run_ExtendableBuffer():
    rb = ReplayBuffer(enable_extend=True)
    print(type(rb))

    for i in range(10):
        rb.store(time=i, x=np.random.rand(4)).flush()

    # print(rb._buf["time"].shape)
    # print(rb._buf["time"][[1, 2, 3], :].shape)
    # exit()

    for _ in range(5):
        batch = rb.sample(2)
        print("-"*30)
        print(batch.time)
        print(batch.time.shape)


if __name__ == '__main__':
    run_ExtendableBuffer()
