import numpy as np


def boltzman_policy(values_of_actions):
    e = np.exp(values_of_actions)
    p = e / e.sum()
    return np.random.multinomial(1, p).argmax()


def epsilon_greedy_policy(values_of_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, values_of_actions.shape[0])
    else:
        return values_of_actions.argmax()
