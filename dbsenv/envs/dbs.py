"""
Environment for simulating Deep Brain Stimulation (DBS) effects on a spiking
neural network.
"""
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

from abc import ABC, abstractmethod
from typing import Optional, Any, SupportsFloat
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dbsenv.utils.neural_model import NeuralModel
from dbsenv.utils.sim_config import SimConfig

ObsType = np.ndarray
ActType = np.ndarray


class DBSEnv(gym.Env, ABC):
    """
    Abstract base environment for DBS simulations.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        sim_config: SimConfig,
        model_class: type[NeuralModel],
        model_params: Optional[dict] = None,
        render_mode: Optional[str] = None
    ) -> None:
        super().__init__()

        self.sim_config = sim_config
        self.model_class = model_class
        self.model_params = model_params or {}
        self.model = model_class(sim_config, **self.model_params)

        # Define the Observation and Action Spaces.
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 20.0, 5.0, 20.0, 5.0]),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float64,
        )

        self.render_mode = render_mode
        self.state = None
        self.i = 1  # sample index

    def _get_obs(self) -> ObsType:
        """
        Returns the current state/observation.
        """

        if self.state is None:
            self.state = np.zeros(
                self.observation_space.shape[0],
                dtype=np.float64
            )

        return np.array(self.state, dtype=np.float64)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self.model = self.model_class(self.sim_config, **self.model_params)
        self.state = None
        self.i = 1

        return self._get_obs(), {}

    @abstractmethod
    def step(
            self,
            action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        raise NotImplementedError

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            if self.state is None:
                raise RuntimeError("Environment has not been reset yet.")
            print(f"Step {self.i}: {self.state}")
        elif mode == "rgb_array":
            raise NotImplementedError(
                "Rendering to RGB array is not implemented."
            )
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def close(self) -> None:
        pass
