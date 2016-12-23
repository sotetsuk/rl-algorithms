import gym
import numpy as np

from rl_algorithms import boltzman_policy, TabularQLearninig


# Set environment
env = gym.make('FrozenLake-v0')
NUM_EPISODE = 10000

# Set algorithm
_lambda = 0.5
tabular_q_learning = TabularQLearninig(env.observation_space, env.action_space, discount_rate=0.99)

# Estimate value function
for n_trial in range(NUM_EPISODE):
    observation = env.reset()
    states = []
    rewards = []
    while True:
        action = boltzman_policy(observation, tabular_q_learning.action_values)
        next_observation, reward, done, _ = env.step(action)
        if observation is not None:
            tabular_q_learning.update(observation, action, reward, next_observation)
        observation = next_observation
        if done:
            break

action_values = tabular_q_learning.action_values
print(action_values)
values = action_values.max(axis=1).reshape((4, 4))
normalized_values = values / values.max()
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print(normalized_values)

