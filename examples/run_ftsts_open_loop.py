"""
Using the FTSTS DBS environment with a fixed action to simulate the original open-loop regime.
"""

import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from dbsenv.neural_models import EILIFNetwork
from dbsenv.neural_models.synchrony import kop
from dbsenv.utils import SimConfig


def plot_kop(t, re):
    """Plot the Kuramoto (Synchrony) Order Parameter."""

    plt.figure(figsize=(10, 5))
    plt.plot(t / 1000, re, label="Kuramoto Order Parameter", color="blue")
    plt.title("Time Series Kuramoto Order Parameter", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("KOP", fontsize=12)
    plt.xlim([0.0, 4.9])
    plt.ylim([0, 1.1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def main():
    """Simulates the original open-loop ftsts regime."""

    # Configure simulation.
    sim_config = SimConfig(
        # duration=25*1000,
        duration=5000,
        step_size=0.1,
        sample_duration=20,
    )

    # Create the environment.
    env = gym.make(
        'dbsenv/DBS-FTSTS-v0',
        sim_config=sim_config,
        model_class=EILIFNetwork,
        model_params={
            "num_e": 160,
            "num_i": 40,
        },
    )
    raw_env = env.unwrapped
    num_samples = raw_env.model.num_samples

    env.reset()
    done = False
    sptime = None
    with tqdm(total=num_samples, desc="Simulating") as pbar:
        while not done:
            action = np.array([100], dtype=np.float64)
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            sptime = info["spike_time_e"]
            pbar.update(1)

    step_size = raw_env.sim_config.step_size
    duration = raw_env.sim_config.duration
    ne = raw_env.model.num_e
    t = np.arange(0.1, duration + step_size, step_size)

    print("computing kop")
    re = kop(
        sptime=sptime,
        t=t,
        step_size=step_size,
        duration=duration,
        num_neurons=ne,
    )

    print("plotting kop")
    plot_kop(t, re)

    env.close()


if __name__ == '__main__':
    main()
