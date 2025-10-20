"""
Environment for simulating Deep Brain Stimulation (DBS) effects on a spiking
neural network.
"""
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

from typing import Optional, Any, SupportsFloat
import numpy as np
from gymnasium import spaces
from dbsenv.utils.neural_model import NeuralModel
from dbsenv.utils.sim_config import SimConfig
from dbsenv.envs.dbs import DBSEnv

ObsType = np.ndarray
ActType = np.ndarray

INHIBITION_THRESHOLD = 75  # (mV)
STIMULATION_ONSET_TIME_RATIO = 0.08  # begin after 8% of the simulation time
PLASTICITY_ONSET_TIME_RATIO = 0.004  # begin after 0.4% of the simulation time
# todo: sample duration ratio
MIN_VSTIM = 10  # (mV)
MAX_VSTIM = 200  # (mV)


class FTSTSEnv(DBSEnv):
    """
    TODO: docstring
    """

    def __init__(
        self,
        sim_config: SimConfig,
        model_class: type[NeuralModel],
        model_params: dict | None = None,
        render_mode: Optional[str] = None
    ) -> None:
        super().__init__(sim_config=sim_config, model_class=NeuralModel)

        self.sim_config = sim_config
        self.model_class = model_class
        self.model_params = model_params or {}
        self.model = model_class(sim_config, **self.model_params)

        self.stim_onset_time = int(
            self.model.duration * STIMULATION_ONSET_TIME_RATIO
        )  # (ms) time when stimulation starts
        self.plasticity_onset_time = int(
            self.model.duration * PLASTICITY_ONSET_TIME_RATIO
        )  # (ms) time when plasticity starts

        # Define the Observation and Action Spaces.
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 20.0, 5.0, 20.0, 5.0]),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=np.array([MIN_VSTIM]),  # (mV) min Vstim
            high=np.array([MAX_VSTIM]),  # (mV) max Vstim
            dtype=np.float64,
        )

        self.render_mode = render_mode
        self.state = None
        self.i = 1  # sample index

    def _get_obs(self) -> ObsType:
        if self.state is None:
            num_state_variables = self.observation_space.shape[0]
            self.state = np.zeros(num_state_variables, dtype=np.float64)

        # todo: use wrapper for bounds on observation space
        (
            synchrony,
            mean_vE,
            std_vE,
            mean_vI,
            std_vI,
        ) = self.state
        assert 0 <= synchrony <= 1
        assert 0 <= mean_vE <= 20
        assert 0 <= std_vE <= 5
        assert 0 <= mean_vI <= 20
        assert 0 <= std_vI <= 5

        return np.array(self.state, dtype=np.float64)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self.model = self.model_class(self.sim_config, **self.model_params)
        self.state = None
        self.i = 1

        self.stim_onset_time = int(
            self.model.duration * STIMULATION_ONSET_TIME_RATIO
        )
        self.plasticity_onset_time = int(
            self.model.duration * PLASTICITY_ONSET_TIME_RATIO
        )

        return self._get_obs(), {}

    def step(
            self,
            action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), \
            f"{action!r} ({type(action)}) invalid."
        assert 1 <= self.i <= self.model.num_samples, \
            f"step {self.i} out of range [1, {self.model.num_samples}]"

        print(self.i)

        sample_duration = self.model.sample_duration  # (ms)
        num_steps_per_sample = self.model.num_steps_per_sample

        self.model.comp_time = (self.i - 1) * sample_duration

        # Calculate Average Effective Inhibition.
        if np.mean(self.model.w_ei0) * self.model.j_i < INHIBITION_THRESHOLD:
            self.model.cross_100 = 0

        # Control Stimulation and Plasticity.
        stim_on = (
            ((self.i * sample_duration) >= self.stim_onset_time)
            * self.model.cross_100
        )
        plast_on = (
            (self.i * sample_duration) >= self.plasticity_onset_time
        )
        assert stim_on in (0, 1)
        assert plast_on in (0, 1)

        # Define the Sample Window.
        sample_start = (self.i >= 2) * (self.i - 1) * num_steps_per_sample
        sample_end = self.i * num_steps_per_sample
        assert sample_end - sample_start == num_steps_per_sample

        # Define the Stimulation Input.
        Vstim = float(action[0])  # (mV) stimulation amplitude
        assert MIN_VSTIM <= Vstim <= MAX_VSTIM
        ue = Vstim * self.model.u_e[:, sample_start:sample_end]
        ui = Vstim * self.model.u_i[:, sample_start:sample_end]
        assert ue.shape[1] == num_steps_per_sample
        assert ui.shape[1] == num_steps_per_sample

        # Run the ODE Neuron Model.
        (
            step_times,
            v_e,
            v_i,
            s_ei,
            s_ie,
            x_ei,
            x_ie,
            apost,
            apre,
            w_ie,
            spike_e,
            spike_i,
            ref_e,
            ref_i,
            synchrony,
            spt_e,
            phif,
        ) = self.model.step(
            ue=ue,
            ui=ui,
            plast_on=plast_on,
            stim_on=stim_on,
            percent_V_stim=1,
        )

        # Recorded (state) Variables.
        mean_vE = np.mean(v_e[0:-1, :], axis=1)
        mean_vI = np.mean(v_i[0:-1, :], axis=1)
        self.model.time_array[sample_start:sample_end, 0] = (
            step_times[0:-1] + (self.i - 1) * sample_duration
        )
        self.model.w_ie[sample_start:sample_end, 0] = np.mean(
            w_ie[0:-1, :],
            axis=1
        )
        self.model.spike_e[sample_start:sample_end, :] = spike_e[0:-1, :]
        self.model.spike_i[sample_start:sample_end, :] = spike_i[0:-1, :]
        self.model.spike_time_e[sample_start:sample_end, :] = spt_e[0:-1, :]

        self.state = (
            # Synchrony.
            synchrony,
            # Voltage Traces.
            np.mean(mean_vE),
            np.std(mean_vE),
            np.mean(mean_vI),
            np.std(mean_vI),

            # etc.
        )

        # Update Initial Conditions (for next run).
        self.model.v_e0 = v_e[-1, :].reshape(1, -1)
        self.model.v_i0 = v_i[-1, :].reshape(1, -1)
        self.model.s_ei0 = s_ei[-1, :].reshape(1, -1)
        self.model.s_ie0 = s_ie[-1, :].reshape(1, -1)
        self.model.x_ei0 = x_ei[-1, :].reshape(1, -1)
        self.model.x_ie0 = x_ie[-1, :].reshape(1, -1)
        self.model.apost0 = apost[-1, :].reshape(1, -1)
        self.model.apre0 = apre[-1, :].reshape(1, -1)
        self.model.w_ie0 = w_ie[-1, :].reshape(1, -1)
        self.model.w_ei0 = self.model.weight_0
        offset = num_steps_per_sample - int(5 / self.model.step_size)
        self.model.leftover_s_ei = s_ei[offset:-1, :]
        self.model.leftover_s_ie = s_ie[offset:-1, :]
        self.model.spt_e0 = spt_e[-1, :]

        state = self._get_obs()
        reward = -synchrony  # todo: define reward function
        self.i += 1
        terminated = self.i > self.model.num_samples
        truncated = False
        infos = {
            "spike_time_e": self.model.spike_time_e,
        }

        if self.render_mode == "human":
            self.render(mode="human")

        return state, reward, terminated, truncated, infos
