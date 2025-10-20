# NeuroLoop DBS Environment

_Closed-Loop Deep-Brain Stimulation for Controlling Synchronization of Spiking Neurons._

### Environments

- `DBSEnv`: Implementation of Deep Brain Stimulation environment.

### Wrappers

This repository hosts the examples that are shown [on wrapper documentation](https://gymnasium.farama.org/api/wrappers/).

- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range
- `DiscreteActions`: An `ActionWrapper` that restricts the action space to a finite subset
- `RelativePosition`: An `ObservationWrapper` that computes the relative position between an agent and a target
- `ReacherRewardWrapper`: Allow us to weight the reward terms for the reacher environment

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
