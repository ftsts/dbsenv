from abc import ABC, abstractmethod


class NeuralModel(ABC):
    """
    Computational Model of the Brain for Simulating Deep Brain Stimulation (DBS).
    """

    # todo: implement
    # @abstractmethod
    # def reset(self):
    #     """Reset the model's internal state."""

    @abstractmethod
    def step(self):
        """Advance the model's dynamics by one time step."""
