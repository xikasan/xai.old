# coding: utf-8


class Agent:

    def __init__(self):
        self.model = None
        self.initialized = False

    def initialize(self):
        raise NotImplementedError()

    def policy(self, state):
        raise NotImplementedError()

    def train(self, batch):
        raise NotImplementedError()
