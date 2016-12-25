import numpy as np
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion

import gym


def rbf_kernels(env, n_samples=100000, gamma=[0.01, 0.1], n_components=100):
    """Represent observation samples using RBF-kernels.

    EXAMPLE
    -------
    >>> env = gym.make('MountainCar-v0')
    >>> n_params, rbf = rbf_kernels(env, n_components=100)
    >>> sample = env.observation_space.sample().reshape((1, env.observation_space.shape[0]))
    >>> rbf(sample).shape
    (1, 100)
    """
    observation_examples = np.array([env.observation_space.sample() for _ in range(n_samples)])

    # Fit feature scaler
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)

    # Fir feature extractor
    features = []
    for g in gamma:
        features.append(('gamma={}'.format(g), RBFSampler(n_components=n_components // len(gamma), gamma=g)))

    features = FeatureUnion(features)
    features.fit(scaler.transform(observation_examples))

    def _rbf_kernels(observation):
        return features.transform(scaler.transform(observation))

    return _rbf_kernels
