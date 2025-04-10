from gymnasium.envs.registration import register

register(
    id="dbs_env/DBS-v0",
    entry_point="dbs_env.envs:DBSEnv",
)
