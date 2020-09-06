# coding: utf-8

import numpy as np
import xtools as xt
import tensorflow as tf
import tensorflow.keras as tk


class TestPolicy(tk.Model):

    def __init__(self):
        super().__init__()

        self.lh1 = tk.layers.Dense(8, name="Lh1")
        self.lo1 = tk.layers.Dense(1, name="Lo1")
        self.lo2 = tk.layers.Dense(2, name="Lo2")

    def call(self, inputs):
        feature = self.lh1(inputs)
        output_1 = self.lo1(feature)
        output_2 = self.lo2(feature)
        return output_1, output_2


if __name__ == '__main__':
    model = TestPolicy()
    dummy_input = np.random.rand(3, 4).astype(np.float32)
    xt.debug("dummy_input")
    print(dummy_input)

    output_1, output_2 = model(dummy_input)
    xt.debug("output 1")
    print(output_1)
    xt.debug("output 2")
    print(output_2)
