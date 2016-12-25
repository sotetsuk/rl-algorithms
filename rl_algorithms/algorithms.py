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


class EveryVisitMC(object):

    def __init__(self, observation_space, action_space, discount_rate=0.01, step_size=0.01):
        self.observation_space = observation_space
        self.action_space = action_space
        self.values = np.zeros(self.observation_space.n, np.float64)  # uniform initialization (zero)
        self.discount_rate = discount_rate
        self.step_size = step_size

    def update(self, states, rewards):
        assert len(states) == len(rewards)
        T = len(states)
        targets = np.zeros_like(self.values, np.float64)

        _sum = 0
        for t in reversed(range(T)):
            _sum += rewards[t] + self.discount_rate * _sum
            targets[states[t]] = _sum
            self.values[states[t]] += self.step_size * (targets[states[t]] - self.values[states[t]])


class TabularTDLambda(object):

    def __init__(self, _lambda, observation_space, action_space, discount_rate=0.01, step_size=0.01):
        self.observation_space = observation_space
        self.action_space = action_space
        self.values = np.zeros(self.observation_space.n, np.float64)  # uniform initialization (zero)
        self.discount_rate = discount_rate
        self.step_size = step_size

        self._lambda = _lambda
        self.eligibility_traces = np.zeros_like(self.values, np.float64)

    def update(self, state, reward, next_state):
        delta = reward + self.discount_rate * self.values[next_state] - self.values[state]
        for x in range(self.observation_space.n):
            self.eligibility_traces[x] = self.discount_rate * self._lambda * self.eligibility_traces[x]
            if state == x:
                self.eligibility_traces[x] = 1
            self.values[x] += self.step_size * delta * self.eligibility_traces[x]


class TabularQLearninig(object):

    def __init__(self, observation_space, action_space, discount_rate=0.01, step_size=0.01):
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_values = np.zeros((self.observation_space.n, self.action_space.n), np.float64)
        self.discount_rate = discount_rate
        self.step_size = step_size

    def update(self, state, action, reward, next_state):
        delta = reward + self.discount_rate * self.action_values[next_state, :].max() - self.action_values[state, action]
        self.action_values[state, action] += self.step_size * delta


class QLearningLinFApp(object):

    def __init__(self, n_params, observation_space, action_space, discount_rate=0.01, step_size=0.01):
        self.observation_space = observation_space
        self.action_space = action_space
        self.discount_rate = discount_rate
        self.step_size = step_size

        self.theta = np.zeros(n_params, np.float64)
        self.phi = None

    def update(self, state, action, reward, next_state):
        phi_per_action = self.phi(next_state, None)
        next_q_max = np.multiply(self.theta.reshape(phi_per_action.shape), phi_per_action).sum(axis=1).max()

        delta = reward + self.discount_rate * next_q_max - np.dot(self.theta, self.phi(state, action))
        self.theta += self.step_size * delta * self.phi(state, action)
