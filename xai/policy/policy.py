# coding: utf-8

import tensorflow.keras as tk
from ..model.mlp import MLP


class Policy(tk.Model):

    def __init__(self, units, dim_action, hidden_activation="relu", name="policy"):
        super().__init__(name=name)

        self.model = MLP(units, activation=hidden_activation)
        self.dim_action = dim_action

    def call(self, state):
        raise NotImplementedError()
