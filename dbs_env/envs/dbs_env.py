from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DBSEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # We have 3 actions (Vstim, Tstim, and x_neutral).
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),
            high=np.array([5.0, 100.0, 1.0]),
            dtype=np.float64
        )  # Vstim: [0, 5], Tstim: [0, 100], x_neutral: [-1, 1]

        # We observe the Kuramoto Order Parameter (KOP) for synchrony.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float64
        )

        # Internal state
        self.state = None
        self.current_step = 0
        self.max_steps = 1  # non-adaptive cl-dbs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.state = np.array([0], dtype=np.float64)  # placeholder state
        self.current_step = 0

        return self.state, {}

    def step(self, action):
        self.current_step += 1

        vstim, tstim, xneutral = action

        # Generate random KOP value.
        kop = np.clip(np.random.normal(loc=0.5 - 0.1 * vstim, scale=0.1), 0, 1)
        energy_penalty = 0.01 * (vstim ** 2)
        reward = -kop - energy_penalty

        self.state = np.array([kop], dtype=np.float64)

        # Episode ends after one step since it's not adaptive dbs.
        terminated = True
        truncated = False

        return self.state, reward, terminated, truncated, {}
