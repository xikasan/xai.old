# coding: utf-8

import tensorflow as tf
import tensorflow.keras as tk


class MLP(tk.Model):

    def __init__(self, units, activation="relu", name="mlp"):
        super().__init__(name=name)

        self.model = tk.Sequential([
            tk.layers.Dense(unit, activation=activation, name="L{}".format(l))
            for l, unit in enumerate(units)
        ])

    def call(self, inputs):
        return self.model(inputs)
