# <img src="./docs/images/ftsts-logo.png" alt="Logo" width="50" style="vertical-align: middle; margin-right: 10px; border-radius: 5px"> dbsenv

_A collection of gymnasium-style environments for modeling Deep Brain Stimulation._

[docs](https://owenmastropietro.github.io/projects/neuroloop/)

---

## Usage

> See [neuroloop](https://owenmastropietro.github.io/projects/neuroloop/) for a practical use case.

### Example

```py
import gymnasium as gym
from dbsenv.envs import DBSEnv
from dbsenv.neural_models import NeuralModel

env = gym.make(
    'dbsenv/DBS-v0',
    sim_config=sim_config,
    model_class=NeuralModel,
    model_params={...},
)
```

> Note: to use the [faster C implementation](./csrc/kop.c) of the Kuramoto Order Paremter to compute synchrony, you must manually compile it.
>
> e.g., `gcc -O3 -fPIC -shared csrc/kop.c -o build/libkuramoto.so`

```py
from dbsenv.utils.synchrony import kop

re = kop(sptime, t, step_size, duration, num_neurons)
```

## Environments

- `DBSEnv`: Abstract Base Class template for Deep Brain Stimulation environments.
- `FTSTSEnv`: Implementation of the Forced Temporal-Spike Time Stimulation environment.

## Wrappers

- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range
- `DiscreteActions`: An `ActionWrapper` that restricts the action space to a finite subset
- `RelativePosition`: An `ObservationWrapper` that computes the relative position between an agent and a target
- `ReacherRewardWrapper`: Allow us to weight the reward terms for the reacher environment
- `DBSNormalizeObservation`: An `ObservationWrapper` that clips and normalizes the observation space

## Contributing

If you would like to contribute, follow these steps:

- Fork this repository
- Clone your fork
- Set up pre-commit via `pre-commit install`
- Make changes, add tests, and submit a pull request
