"""
TODO: docstring
"""
# pylint: disable=invalid-name

from typing import Optional, Any, SupportsFloat
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .ode_neuron_model import ode_neuron_model
from .dbs_utils import make_synaptic_connections, pulsatile_input
from .neural_model import NeuralModel

ObsType = np.ndarray
ActType = np.ndarray

INHIBITION_THRESHOLD = 75  # (mV)
STIMULATION_ONSET_TIME_RATIO = 0.08  # begin after 8% of the simulation time
PLASTICITY_ONSET_TIME_RATIO = 0.004  # begin after 0.4% of the simulation time
# todo: sample duration ratio


class DBSEnv(gym.Env):
    """
    TODO: docstring
    """

    def __init__(self):
        super().__init__()

        # Run Parameters.
        self.duration = 5_000  # (ms) duration of the simulation
        self.step_size = 0.1  # (ms) time step of the simulation
        self.num_steps = int(self.duration / self.step_size)
        self.sample_duration = 20
        self.num_samples = int(self.duration / self.sample_duration)
        self.num_steps_per_sample = int(self.sample_duration / self.step_size)
        self.stim_onset_time = int(
            self.duration * STIMULATION_ONSET_TIME_RATIO
        )  # (ms) time when stimulation starts
        self.plasticity_onset_time = int(
            self.duration * PLASTICITY_ONSET_TIME_RATIO
        )  # (ms) time when plasticity starts

        # Neuron Parameters.
        self.mew_e = 20.8
        self.sigma_e = 1
        self.mew_i = 18
        self.sigma_i = 3
        self.mew_c = 0
        self.num_e = 160
        self.num_i = 40
        self.num_neurons = self.num_e + self.num_i

        # Synaptic Parameters.
        self.weight_0 = 1
        self.j_e = 100  # synaptic strength (J_E = J_EI)
        self.j_i = 260  # synaptic strength (J_I = J_IE)
        self.n_i = 1  # copilot: number of synapses per neuron?
        self.c_e = 0.3 * self.num_neurons  # N_I;
        self.c_i = 0.3 * self.num_neurons  # N_E;
        self.tau_ltp = 20  # long-term potentiation time constant
        self.tau_ltd = 22  # long-term depression time constant

        self.epsilon_e = 0.1
        self.epsilon_i = 0.1

        # Make Random Synaptic Conncetions.
        self.s_key_ei, self.num_synapses_ei = make_synaptic_connections(  # I -> E
            num_pre=self.num_i,
            num_post=self.num_e,
            epsilon=self.epsilon_i,
        )
        self.s_key_ie, self.num_synapses_ie = make_synaptic_connections(  # E -> I
            num_pre=self.num_e,
            num_post=self.num_i,
            epsilon=self.epsilon_e,
        )

        self.w_ie = np.zeros((self.num_steps, 1))
        self.w_ie_std = np.zeros((int(self.num_steps), 1))

        # Initial Conditions (todo: state?).
        self.v_e0 = 14 * np.ones((1, self.num_e))
        self.v_i0 = 14 * np.ones((1, self.num_i))
        self.s_ei0 = np.zeros((1, self.num_e))
        self.s_ie0 = np.zeros((1, self.num_i))
        self.x_ei0 = np.zeros((1, self.num_e))
        self.x_ie0 = np.zeros((1, self.num_i))
        self.apost0 = np.zeros((1, int(self.num_synapses_ie)))
        self.apre0 = np.zeros((1, int(self.num_synapses_ie)))
        self.w_ei0 = self.weight_0
        self.w_ie0 = self.weight_0 * np.ones((1, int(self.num_synapses_ie)))
        self.leftover_s_ei = np.zeros((
            int(5 / self.step_size) + 1,
            self.num_e
        ))
        self.leftover_s_ie = np.zeros((
            int(5 / self.step_size) + 1,
            self.num_i
        ))
        self.ref_e = np.zeros((1, self.num_e))
        self.ref_i = np.zeros((1, self.num_i))
        self.spt_e0 = 0
        self.sp_count_e0 = 0
        self.phi0 = np.zeros((1, self.num_e))
        self.phif = np.zeros((1, self.num_e))

        self.synchrony = np.zeros((int(self.num_samples), 1))
        self.time_syn = np.zeros((int(self.num_samples), 1))
        self.spike_time_e = np.zeros((self.num_steps, self.num_e))
        self.sp_count_e = np.zeros((
            int(self.num_steps_per_sample),
            self.num_e
        ))
        self.tau_e_m = np.full((1, self.num_e), 10)
        self.tau_i_m = np.full((1, self.num_i), 10)

        # Generate General Stimulation Pattern
        self.cross_100 = 1
        self.comp_time = 0
        self.v_stim = 1
        self.t_stim = 1
        self.x_neutral = 10
        self.multiple = 1
        self.u_e, self.u_i = pulsatile_input(
            multi=self.multiple,
            v_stim=self.v_stim,
            t_stim=self.t_stim,
            x=self.x_neutral,
            duration=self.duration,
            step_size=self.step_size,
        )
        self.u_e = self.u_e.reshape(1, -1)  # (N,) -> (1, N)
        self.u_i = self.u_i.reshape(1, -1)  # (N,) -> (1, N)

        self.time_array = np.zeros((self.num_steps, 1))

        # Just synchrony for now.
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 20.0, 5.0, 20.0, 5.0]),
            dtype=np.float64
        )

        # Just stimulation amplitude for now.
        self.action_space = spaces.Box(
            low=np.array([1000.0]),  # (mV) min Vstim
            high=np.array([4000.0]),  # (mV) max Vstim
            dtype=np.float64,
        )

        # todo: what to do with you
        self.voltage_traces = 0, 0, 0, 0

        self.state = None
        self.i = 0

    def _get_obs(self) -> ObsType:
        """
        Returns the current state/observation.
        """

        # Observe Synchrony.
        synchrony = self.synchrony[self.i - 1, 0]
        assert 0 <= synchrony <= 1

        # Observe Voltage Traces.
        mean_vE, std_vE, mean_vI, std_vI = self.voltage_traces
        assert 0 <= mean_vE <= 20
        assert 0 <= std_vE <= 5
        assert 0 <= mean_vI <= 20
        assert 0 <= std_vI <= 5

        return np.array(
            [
                synchrony,
                mean_vE,
                std_vE,
                mean_vI,
                std_vI,
            ],
            dtype=np.float64
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Resets some stuff...
        """

        super().reset(seed=seed)
        # what changes, and needs to be reset?
        # - all initial conditions
        # - time_array
        # - W_EI
        # - Synchrony (covered in init. cond.)
        # - time_syn (covered in init. cond.)
        # - spike_time_E (covered in init. cond.)
        # - s_key_EI, num_synapses_EI
        # - s_key_IE, num_synapses_IE
        # - W_IE and W_IE_std ?

        self.i = 1  # [1, num_samples + 1]
        self.cross_100 = 1

        # Initial Conditions.
        self.v_e0 = 14 * np.ones((1, self.num_e))
        self.v_i0 = 14 * np.ones((1, self.num_i))
        self.s_ei0 = np.zeros((1, self.num_e))
        self.s_ie0 = np.zeros((1, self.num_i))
        self.x_ei0 = np.zeros((1, self.num_e))
        self.x_ie0 = np.zeros((1, self.num_i))
        self.apost0 = np.zeros((1, int(self.num_synapses_ie)))
        self.apre0 = np.zeros((1, int(self.num_synapses_ie)))
        self.w_ie0 = self.weight_0 * np.ones((1, int(self.num_synapses_ie)))
        self.w_ei0 = self.weight_0
        self.leftover_s_ei = np.zeros((
            int(5 / self.step_size) + 1,
            self.num_e
        ))
        self.leftover_s_ie = np.zeros((
            int(5 / self.step_size) + 1,
            self.num_i
        ))
        self.ref_e = np.zeros((1, self.num_e))
        self.ref_i = np.zeros((1, self.num_i))
        self.spt_e0 = 0
        self.sp_count_e0 = 0
        self.phi0 = np.zeros((1, self.num_e))  # todo: unused...
        self.phif = np.zeros((1, self.num_e))

        self.num_samples = int(self.duration / self.sample_duration)
        self.synchrony = np.zeros((int(self.num_samples), 1))
        self.time_syn = np.zeros((int(self.num_samples), 1))
        self.num_steps_per_sample = int(self.sample_duration / self.step_size)
        self.spike_time_e = np.zeros((self.num_steps, self.num_e))
        self.sp_count_e = np.zeros((
            int(self.num_steps_per_sample),
            self.num_e
        ))
        self.tau_e_m = np.full((1, self.num_e), 10)
        self.tau_i_m = np.full((1, self.num_i), 10)

        return self._get_obs(), {}

    def step(
            self,
            action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Takes a step (runs one sample of ODE).
        """
        assert isinstance(action, np.ndarray)

        self.comp_time = (self.i - 1) * self.sample_duration

        # Calculate Average Effective Inhibition.
        if np.mean(self.w_ei0) * self.j_i < INHIBITION_THRESHOLD:
            self.cross_100 = 0

        # Control Input and Plasticity.
        stim_on = (
            ((self.i * self.sample_duration) >= self.stim_onset_time)
            * self.cross_100
        )
        plast_on = (
            (self.i * self.sample_duration) >= self.plasticity_onset_time
        )

        # Define the Sample Window.
        sample_start = (self.i >= 2) * (self.i - 1) * self.num_steps_per_sample
        sample_end = self.i * self.num_steps_per_sample
        assert sample_end - sample_start == self.num_steps_per_sample

        # Define the Stimulation Input.
        Vstim = float(action[0])  # (mV) stimulation amplitude
        assert 1000 <= Vstim <= 4000
        ue = Vstim * self.u_e[:, sample_start:sample_end]
        ui = Vstim * self.u_i[:, sample_start:sample_end]
        assert ue.shape[1] == self.num_steps_per_sample
        assert ui.shape[1] == self.num_steps_per_sample

        percent_V_stim = 1
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
        ) = ode_neuron_model(
            plast_on=plast_on,
            ON=stim_on,
            vE0=self.v_e0,
            vI0=self.v_i0,
            S_EI0=self.s_ei0,
            S_IE0=self.s_ie0,
            X_EI0=self.x_ei0,
            X_IE0=self.x_ie0,
            Apost0=self.apost0,
            Apre0=self.apre0,
            W_IE0=self.w_ie0,
            W_EI0=self.w_ei0,
            mew_e=self.mew_e,
            sigma_e=self.sigma_e,
            ue=ue,
            ui=ui,
            mew_i=self.mew_i,
            sigma_i=self.sigma_i,
            J_E=self.j_e,
            J_I=self.j_i,
            C_E=self.c_e,
            C_I=self.c_i,
            tau_LTP=self.tau_ltp,
            tau_LTD=self.tau_ltd,
            step_size=self.step_size,
            sample_duration=self.sample_duration,
            N_E=self.num_e,
            N_I=self.num_i,
            S_key_EI=self.s_key_ei,
            S_key_IE=self.s_key_ie,
            leftover_S_EI=self.leftover_s_ei,
            leftover_S_IE=self.leftover_s_ie,
            ref_E=self.ref_e,
            ref_I=self.ref_i,
            tau_E_m=self.tau_e_m,
            tau_I_m=self.tau_i_m,
            percent_V_stim=percent_V_stim,
            comp_time=self.comp_time,
            spt_E0=self.spt_e0,
            phif=self.phif,
        )

        # Recorded Variables
        self.time_array[sample_start:sample_end, 0] = (
            step_times[0:-1, 0] + (self.i - 1) * self.sample_duration
        )
        self.w_ie[sample_start:sample_end, 0] = np.mean(w_ie[0:-1, :], axis=1)
        self.synchrony[self.i - 1, 0] = synchrony
        self.time_syn[self.i - 1, 0] = self.sample_duration * self.i
        self.spike_time_e[sample_start:sample_end, :] = spt_e[0:-1, :]

        # todo: for get_obs, but need to compute here :(
        mean_vE = np.mean(v_e[0:-1, :], axis=1)
        mean_vI = np.mean(v_i[0:-1, :], axis=1)
        self.voltage_traces = (
            np.mean(mean_vE),
            np.std(mean_vE),
            np.mean(mean_vI),
            np.std(mean_vI),
        )

        # Generate Initial Condition (for next run).
        self.v_e0 = v_e[-1, :].reshape(1, -1)
        self.v_i0 = v_i[-1, :].reshape(1, -1)
        self.s_ei0 = s_ei[-1, :].reshape(1, -1)
        self.s_ie0 = s_ie[-1, :].reshape(1, -1)
        self.x_ei0 = x_ei[-1, :].reshape(1, -1)
        self.x_ie0 = x_ie[-1, :].reshape(1, -1)
        self.apost0 = apost[-1, :].reshape(1, -1)
        self.apre0 = apre[-1, :].reshape(1, -1)
        self.w_ie0 = w_ie[-1, :].reshape(1, -1)
        self.w_ei0 = self.weight_0
        left_sample_end = self.num_steps_per_sample - int(5 / self.step_size)
        self.leftover_s_ei = s_ei[left_sample_end:-1, :]
        self.leftover_s_ie = s_ie[left_sample_end:-1, :]
        self.spt_e0 = spt_e[-1, :]

        state = self._get_obs()
        reward = -synchrony
        self.i += 1
        terminated = self.i >= self.num_samples
        truncated = False
        infos = {}

        return state, reward, terminated, truncated, infos
