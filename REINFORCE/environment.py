import gym

import numpy as np


class Environment:

    def __init__(self, config):
        self.config = config
        self.make()

    def initialize_game(self):
        self._state = self.env.reset()
        return self.state

    def _step(self, action):
        self._state, self.reward, self.done, self.info = self.env.step(action)

    def render(self):
        self.env.render()

    @property
    def state_shape(self):
        return (4,)

    @property
    def action_space_size(self):
        return self.env.action_space.n

    @property
    def state(self):
        return self._state

    def act(self, action):
        self._step(action)
        return self.state, self.reward, self.done

    def make(self):
        self.env = gym.make("CartPole-v0")
        self._state = None
        self.reward = 0
        self.done = True
        self.info = None