# coding: utf-8


class Observer:

    def __init__(self, env):
        self._env = env

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        return self.transform(self._env.reset())

    def render(self):
        self._env.render(mode="human")

    def step(self, action):
        next_state, reward, done, info = self._env.step(action)
        return self.transform(next_state), reward, done, info

    def transform(self, state):
        raise NotImplementedError()
