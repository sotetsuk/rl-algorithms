import gym
import numpy as np

from rl_algorithms import EveryVisitMC


# Set environment
env = gym.make('FrozenLake-v0')
NUM_EPISODE = 100000

# Set algorithm
every_visit_mc = EveryVisitMC(env.observation_space, env.action_space, discount_rate=0.1)

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
            states.append(observation)
            rewards.append(reward)
        observation = next_observation
        if done:
            break
    every_visit_mc.update(states, rewards)


values = every_visit_mc.values.reshape((4, 4))
normalized_values = values / values.max()
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print(normalized_values)
