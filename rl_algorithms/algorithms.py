import numpy as np


class TabularTdZero(object):
    """Tabular TD(0) algorithm. """

    def __init__(self, observation_space, action_space, discount_rate=0.01, step_size=0.01):
        self.observation_space = observation_space
        self.action_space = action_space
        self.values = np.zeros(self.observation_space.n, np.float64)  # uniform initialization (zero)
        self.discount_rate = discount_rate
        self.step_size = step_size

    def update(self, state, reward, next_state):
        """Implement pseudo-code.

        :param state:
        :param reward:
        :param next_state:
        :return:
        """
        delta = reward + self.discount_rate * self.values[next_state] - self.values[state]
        self.values[state] += self.step_size * delta
