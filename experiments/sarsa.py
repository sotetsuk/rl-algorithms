import gym
import numpy as np
from rl_algorithms import rbf_kernels, epsilon_greedy_policy, SARSA

# Set environment
env = gym.make('MountainCar-v0')

# Parameters
NUM_EPISODE = 1000
N_PARAMS = 9 * 100
DISCOUNT_RATE = 1.0
MAX_T = 20000
GAMMA = [0.01, 0.1, 1.0]
EPSILON_INIT = 0.1
EPSILON_DECAY = 0.998
STEP_SIZE = 0.1
STEP_SIZE_DECAY = 0.998
LAMBDA = 0.9


def get_phi(n_params, gamma):
    rbf = rbf_kernels(env, gamma=gamma, n_components=n_params // env.action_space.n)

    def phi(observation, action):
        n_action = env.action_space.n
        phi_vals = rbf(observation.reshape((1, observation.shape[0])))
        phi_vals = np.tile(phi_vals, (n_action, 1))
        if action is None:
            return phi_vals[:, :]

        mask = np.zeros_like(phi_vals)
        mask[action, :] = 1.

        return np.multiply(phi_vals, mask).flatten()

    return phi

# Setting algorithms
sarsa = SARSA(LAMBDA, N_PARAMS, env.observation_space, env.action_space,
                              discount_rate=DISCOUNT_RATE, step_size=STEP_SIZE)
sarsa.phi = get_phi(N_PARAMS, GAMMA)

env.monitor.start('./mountain-car-v0', force=True)
# Estimate value function
epsilon = EPSILON_INIT
for n_trial in range(NUM_EPISODE):
    print("trial: {}".format(n_trial))
    observation = env.reset()
    phi_per_action = sarsa.phi(observation, None)
    values_of_actions = np.multiply(sarsa.theta.reshape(phi_per_action.shape), phi_per_action).sum(axis=1)
    action = epsilon_greedy_policy(values_of_actions, epsilon)

    for t in range(MAX_T):
        # transition
        next_observation, reward, done, _ = env.step(action)
        # define next action
        phi_per_action = sarsa.phi(next_observation, None)
        values_of_actions = np.multiply(sarsa.theta.reshape(phi_per_action.shape), phi_per_action).sum(axis=1)
        next_action = epsilon_greedy_policy(values_of_actions, epsilon)
        # SARSA update
        sarsa.update(observation, action, reward, next_observation, next_action)
        observation = next_observation
        action = next_action

        if done:
            break

    epsilon *= EPSILON_DECAY
    sarsa.step_size *= STEP_SIZE_DECAY

env.monitor.close()
