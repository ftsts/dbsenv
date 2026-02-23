from gymnasium.envs.registration import register
from dbsenv.envs import DBSEnv, FTSTSEnv
from dbsenv.neural_models import NeuralModel, EILIFNetwork
from dbsenv.utils.sim_config import SimConfig


register(
    id="dbsenv/DBS-FTSTS-v0",
    entry_point="dbsenv.envs.ftsts:FTSTSEnv",
)


__all__ = [
    "DBSEnv",
    "FTSTSEnv",
    "NeuralModel",
    "EILIFNetwork",
    "SimConfig",
]
