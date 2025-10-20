from gymnasium.envs.registration import register
from dbs_env.envs.dbs_env import DBSEnv
from dbs_env.utils.sim_config import SimConfig
from dbs_env.utils.neural_model import NeuralModel


register(
    id="dbs_env/DBS-v0",
    entry_point="dbs_env.envs:DBSEnv",
)


__all__ = [
    "DBSEnv",
    "SimulationConfig",
    "NeuralModel",
]
