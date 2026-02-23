from gymnasium.envs.registration import register
from dbsenv.envs.ftsts import FTSTSEnv
from dbsenv.utils.sim_config import SimConfig
from dbsenv.utils.neural_model import NeuralModel


register(
    id="dbsenv/DBS-FTSTS-v0",
    entry_point="dbsenv.envs.ftsts:FTSTSEnv",
)


__all__ = [
    "FTSTSEnv",
    "SimConfig",
    "NeuralModel",
]
