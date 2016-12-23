import gym
import numpy as np

from rl_algorithms import TabularTDLambda


# Set environment
env = gym.make('FrozenLake-v0')
NUM_EPISODE = 10000

# Set algorithm
_lambda = 0.5
tabular_td_lambda = TabularTDLambda(_lambda, env.observation_space, env.action_space, discount_rate=0.99)

# Estimate value function
for n_trial in range(NUM_EPISODE):
    env.reset()
    observation = None
    states = []
    rewards = []
    while True:
        action = env.action_space.sample()  # uniformly randomly sample to estimate value function
        next_observation, reward, done, _ = env.step(action)
        if observation is not None:
            tabular_td_lambda.update(observation, reward, next_observation)
        observation = next_observation
        if done:
            break

values = tabular_td_lambda.values.reshape((4, 4))
normalized_values = values / values.max()
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print(normalized_values)

