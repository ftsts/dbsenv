# <img src="./docs/images/ftsts-logo.png" alt="Logo" width="50" style="vertical-align: middle; margin-right: 10px;"> NeuroLoop DBS Environment

_Closed-Loop Deep-Brain Stimulation for Controlling Synchronization of Spiking Neurons._

### Environments

- `DBSEnv`: Abstract Base Class template for Deep Brain Stimulation environments.
- `FTSTSEnv`: Implementation of the Forced Temporal-Spike Time Stimulation environment.

### Wrappers

This repository hosts the examples that are shown [on wrapper documentation](https://gymnasium.farama.org/api/wrappers/).

- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range
- `DiscreteActions`: An `ActionWrapper` that restricts the action space to a finite subset
- `RelativePosition`: An `ObservationWrapper` that computes the relative position between an agent and a target
- `ReacherRewardWrapper`: Allow us to weight the reward terms for the reacher environment
- `DBSNormalizeObservation`: An `ObservationWrapper` that clips and normalizes the observation space

### Contributing

If you would like to contribute, follow these steps:

- Fork this repository
- Clone your fork
- Set up pre-commit via `pre-commit install`

## Installation

To install your new environment, run the following commands:

```{shell}
cd dbsenv
pip install -e .
```
