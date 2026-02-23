import numpy as np
import gymnasium as gym


class DBSNormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.low = env.observation_space.low
        self.high = env.observation_space.high

    def observation(self, obs):
        _obs = np.clip(obs, self.low, self.high)
        norm = (_obs - self.low) / (self.high - self.low + 1e-8)
        assert 0 <= norm.all() <= 1

        return norm
