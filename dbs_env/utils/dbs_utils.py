# TODO: NUKE THIS TRASH

class SimulationConfig:
    """
    A temporary consolidation of shared parameters until I figure out how
    best to share them.
    """

    def __init__(
            self,
            duration=25_000,  # (ms) duration of simulation
            step_size=0.1,  # (ms)
            sample_duration=20,  # (ms)
            seed=None,
    ):
        self.seed = seed
        self.duration = duration
        self.step_size = step_size
        self.sample_duration = sample_duration
