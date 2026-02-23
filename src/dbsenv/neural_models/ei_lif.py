"""
Neural Model for FTSTS.
"""

import numpy as np
from dbsenv.neural_models.base import NeuralModel
from dbsenv.utils import SimConfig


DEFAULT_NE = 1600  # number of excitatory neurons
DEFAULT_NI = 400  # number of inhibitory neurons

V_REST = 0  # (mV) resting potential
V_THRESHOLD = 20  # (mV) threshold potential for action potential
V_RESET = 14  # (mV) reset potential after action potential
REFRACTORY = 2  # (ms) refractory period after action potential
MAX_WEIGHT_IE = 10  # (mV) maximum synaptic weight for E->I connections
MAX_WEIGHT_EI = 290  # (mV) maximum synaptic weight for I->E connections


class EILIFNetwork(NeuralModel):
    """
    Neural Network for Simulating Deep Brain Stimulation (DBS).

    Excitatory-Inhibitory Network of Leaky-Integrate-and-Fire Neurons with Spike-Timing-Dependent-Plasticity.
    """

    def __init__(self, sim_config: SimConfig, **model_params):
        # self.seed = model_params.get("seed", 42)
        # np.random.seed(self.seed)

        # Run Parameters.
        self.duration = sim_config.duration
        self.step_size = sim_config.step_size
        self.sample_duration = sim_config.sample_duration
        # self.sample_duration = int(duration * SAMPLE_RATIO)
        assert self.sample_duration > 0

        self.num_steps = int(self.duration / self.step_size)
        self.num_samples = int(self.duration / self.sample_duration)
        self.num_steps_per_sample = int(self.sample_duration / self.step_size)

        # Neuron Parameters.
        self.mew_e = 20.8
        self.sigma_e = 1
        self.mew_i = 18
        self.sigma_i = 3
        self.mew_c = 0
        self.num_e = model_params.get("num_e", DEFAULT_NE)
        self.num_i = model_params.get("num_i", DEFAULT_NI)
        self.num_neurons = self.num_e + self.num_i

        # Synaptic Parameters.
        self.weight_0 = 1
        self.j_e = 100  # synaptic strength (J_E = J_EI)
        self.j_i = 260  # synaptic strength (J_I = J_IE)
        self.n_i = 1
        self.c_e = 0.3 * self.num_neurons  # N_I;
        self.c_i = 0.3 * self.num_neurons  # N_E;
        self.tau_ltp = 20  # long-term potentiation time constant
        self.tau_ltd = 22  # long-term depression time constant

        # Make Random Synaptic Connections.
        self.epsilon_e = 0.1
        self.epsilon_i = 0.1
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
        self.sp_e_count0 = 0
        self.phi0 = np.zeros((1, self.num_e))
        self.phif = np.zeros((1, self.num_e))
        self.synchrony = np.zeros((int(self.num_samples), 1))
        # synchrony_index
        self.time_syn = np.zeros((int(self.num_samples), 1))
        self.spike_e = np.zeros((self.num_steps, self.num_e))
        self.spike_i = np.zeros((self.num_steps, self.num_i))
        self.spike_time_e = np.zeros((self.num_steps, self.num_e))
        self.sp_e_count = np.zeros((
            int(self.num_steps_per_sample),
            self.num_e
        ))
        self.tau_e_m = np.full((1, self.num_e), 10)
        self.tau_i_m = np.full((1, self.num_i), 10)

        # Generate General Stimulation Pattern.
        self.cross_100 = 1
        self.comp_time = 0
        self.v_stim = 1  # (mV) stimulation amplitude (todo: rename: Ustim)
        self.t_stim = 1  # (ms)
        self.x_neutral = 10  # (ms) (todo: rename: t_neutral)
        self.multiple = 1
        self.t_pulse = self.t_stim * (self.x_neutral + self.multiple + 1)
        self.u_e, self.u_i = pulsatile_input(  # todo: rename Ve, Vi?
            multi=self.multiple,
            v_stim=self.v_stim,
            t_stim=self.t_stim,
            x=self.x_neutral,
            duration=self.duration,
            step_size=self.step_size
        )
        self.u_e = self.u_e.reshape(1, -1)  # (N,) -> (1, N)
        self.u_i = self.u_i.reshape(1, -1)  # (N,) -> (1, N)
        self.time_array = np.zeros((self.num_steps, 1))

    def _dVdt(self, v, z, mu, sigma, tau_m, x, stim) -> np.ndarray:
        """
        Returns the time derivative of the membrane potential for neurons.

        Equation: τm * dV(t)/dt = -V(t) + Z(t) + μ + (σ * sqrt(τm) * X(t)) + Vstim(t)
        """
        # todo: link to paper/equation

        return (
            V_REST - v + z + mu + sigma * np.sqrt(tau_m) * x + stim
        ) / tau_m

    def step(self, ue, ui, plast_on, stim_on, percent_V_stim=1):
        """
        ODE Neuron Model.

        Function to simulate a network of excitatory and inhibitory neurons
        with synaptic plasticity.
        """

        num_steps = self.num_steps_per_sample

        # Neuron Parameters.
        tau_d = 1
        tau_r = 1
        x_e = np.random.randn(num_steps + 1, self.num_e)  # white noise
        x_i = np.random.randn(num_steps + 1, self.num_i)  # white noise

        # Synaptic Parameters.
        WEI = self.w_ei0
        syn_delay = 5  # (ms)

        # Plasticity Parameters.
        dApre_0 = 0.005*1
        dApost_0 = dApre_0*1
        Apre_i = 0
        Apost_i = 0
        a_LTD = -1 * plast_on * 1.1
        a_LTP = 1 * plast_on * 1
        eta = 0.25

        # Initialize State Vectors.
        # todo: can reduce memory with curr, next instead of full arrays
        # todo: maybe not... np.mean(v_e, axis=1)
        v_e = np.zeros((num_steps + 1, self.num_e))
        v_i = np.zeros((num_steps + 1, self.num_i))
        s_ei = np.zeros((num_steps + 1, self.num_e))
        s_ie = np.zeros((num_steps + 1, self.num_i))
        x_ei = np.zeros((num_steps + 1, self.num_e))
        x_ie = np.zeros((num_steps + 1, self.num_i))
        apost = np.zeros((num_steps + 1, self.num_synapses_ie))
        apre = np.zeros((num_steps + 1, self.num_synapses_ie))
        w_ie = np.zeros((num_steps + 1, self.num_synapses_ie))
        spike_e = np.zeros((num_steps + 1, self.num_e))
        spike_i = np.zeros((num_steps + 1, self.num_i))
        # spt_E = np.zeros((1, self.num_e))
        spt_e = np.zeros((num_steps + 1, self.num_e))
        step_times = np.zeros((num_steps + 1,))

        v_e[0, :] = self.v_e0
        v_i[0, :] = self.v_i0
        s_ei[0, :] = self.s_ei0
        s_ie[0, :] = self.s_ie0
        x_ei[0, :] = self.x_ei0
        x_ie[0, :] = self.x_ie0
        apost[0, :] = self.apost0
        apre[0, :] = self.apre0
        w_ie[0, :] = self.w_ie0
        spt_e[0, :] = self.spt_e0

        # Stimulation Parameters.
        # number of E neurons stimulating
        stim_percent_E = np.zeros(self.num_e)
        a = 0
        b = int(np.floor(percent_V_stim * self.num_e))
        stim_percent_E[a:b] = 1  # TO MAKE FIGURES 3,4,7
        # stim_percent_E[a:b] = 1 + 0.1*np.random.randn(b)  # TO MAKE FIGURE 6

        # number of I neurons stimulating
        stim_percent_I = np.zeros(self.num_i)
        a = 0
        b = int(np.floor(percent_V_stim * self.num_i))
        stim_percent_I[a:b] = 1  # FIGURES 3,4,7
        # stim_percent_I[a:b] = 1 + 0.1*np.random.randn(b)  # FIGURE 6

        spike_I_time = np.zeros((num_steps + 1, self.num_i))
        delay_index = int(syn_delay / self.step_size) - 1

        for t in range(num_steps):
            step_times[t + 1] = step_times[t] + self.step_size

            # Excitatory-Inhibitory Network Model Updates.
            # Voltage (membrane) potentials.
            # (eq1&2): τm dV(t)/dt = -V(t) + Z(t) μ + (σ * sqrt(τm) * X(t)) + Vstim(t)
            # FIGURES 3,4,7
            dv_Edt = self._dVdt(  # equation (1) in paper
                v=v_e[t, :],
                z=self.j_e / self.c_e * (  # equation (3) in paper
                    s_ei[t - delay_index, :] if t > delay_index
                    else self.leftover_s_ei[t, :]
                ),
                mu=self.mew_e,
                stim=stim_on * ue[0, t],
                sigma=self.sigma_e,
                tau_m=self.tau_e_m[0, :],
                x=x_e[t, :],
            )
            dv_Idt = self._dVdt(  # equation (2) in paper
                v=v_i[t, :],
                z=self.j_i / self.c_i * (  # equation (3) in paper
                    s_ie[t - delay_index, :] if t > delay_index
                    else self.leftover_s_ie[t, :]
                ),
                mu=self.mew_i,
                stim=stim_on * ui[0, t],
                sigma=self.sigma_i,
                tau_m=self.tau_i_m[0, :],
                x=x_i[t, :],
            )
            # FIGURE 6
            # dv_Edt[i, :] = (vrest - v_E[i, :] + J_E/C_E * S_EI[i-delay_index, :] + mew_e + ON1 * ue[0, i] + ON1 * 10 * np.random.randn(1, N_E) + sigma_e * (tau_E_m[0, :]**0.5) * whitenoise_E[i, :]) / tau_E_m[0, :]
            # dv_Edt[i, :] = (vrest - v_E[i, :] + J_E/C_E * leftover_S_EI[i, :] + mew_e + ON1 * ue[0, i] + ON1 * 10 * np.random.randn(1, N_E) + sigma_e * (tau_E_m[0, :]**0.5) * whitenoise_E[i, :]) / tau_E_m[0, :]
            # dv_Idt[i, :] = (vrest - v_I[i, :] + J_I/C_I * S_IE[i-delay_index, :] + mew_i + ON1 * ui[0, i] + ON1 * 10 * np.random.randn(1, N_I) + sigma_i * (tau_I_m[0, :]**0.5) * whitenoise_I[i, :]) / tau_I_m[0, :]
            # dv_Idt[i, :] = (vrest - v_I[i, :] + J_I/C_I * leftover_S_IE[i, :] + mew_i + ON1 * ui[0, i] + ON1 * 10 * np.random.randn(1, N_I) + sigma_i * (tau_I_m[0, :]**0.5) * whitenoise_I[i, :]) / tau_I_m[0, :]
            v_e[t + 1, :] = v_e[t, :] + self.step_size * dv_Edt
            v_i[t + 1, :] = v_i[t, :] + self.step_size * dv_Idt

            # Conductance Updates.
            # (eq5): τd dS(t)/dt = -S(t) + X(t)
            dS_EIdt = (  # equation (5) in paper
                -s_ei[t, :] + x_ei[t, :]
            ) / tau_d
            dS_IEdt = (  # equation (5) in paper
                -s_ie[t, :] + x_ie[t, :]
            ) / tau_d
            s_ei[t + 1, :] = s_ei[t, :] + self.step_size * dS_EIdt
            s_ie[t + 1, :] = s_ie[t, :] + self.step_size * dS_IEdt

            # (eq6): τr dX(t)/dt = -X(t) + W(t) * δ(t - tpre + tdelay)
            dX_EIdt = (  # equation (6) in paper
                -x_ei[t, :]  # + W(t) * δ(t - tpre + tdelay) in spike check
            ) / tau_r
            dX_IEdt = (  # equation (6) in paper
                -x_ie[t, :]  # + W(t) * δ(t - tpre + tdelay) in spike check
            ) / tau_r
            x_ei[t + 1, :] = x_ei[t, :] + self.step_size * dX_EIdt
            x_ie[t + 1, :] = x_ie[t, :] + self.step_size * dX_IEdt

            # Spike-Timing Dependence Plasticity (STDP) Updates.
            # (eq7): W(t + Δt) = W(t) + ΔW(t)
            # (eq8): ΔW(t) = η * aLTP * Apost(t) if tpre - tpost < 0
            # (eq9): ΔW(t) = η * aLTD * Apre(t) if tpre - tpost > 0
            w_ie[t + 1, :] = (  # equation (7) in paper
                w_ie[t, :]  # + ΔW(t) in spike check
            )

            # (eq10): τLTP dApost/dt = -Apost + A0 * δ(t - tpost)
            dApostdt = (
                -apost[t, :]  # + A0 * δ(t - tpost) in spike check
            ) / self.tau_ltd
            apost[t + 1, :] = apost[t, :] + self.step_size * dApostdt

            # (eq11): τLTD dApre/dt = -Apre + A0 * δ(t - tpre)
            dApredt = (
                -apre[t, :]  # + A0 * δ(t - tpre) in spike check
            ) / self.tau_ltp
            apre[t + 1, :] = apre[t, :] + self.step_size * dApredt

            # Refractory Updates.
            self.ref_e[0, :] -= self.step_size
            self.ref_i[0, :] -= self.step_size

            # Calculate Kuramoto Order (but not really).
            spt_e[t + 1, :] = spt_e[t, :]
            # phif[i+1+int(comp_time/step), :] = 2*np.pi * (time[i+1, 0] + comp_time - spt_E[i, :])

            # Check for Spikes.
            for e in range(self.num_e):  # excitatory
                if v_e[t, e] < V_THRESHOLD <= v_e[t + 1, e]:
                    # Record Spike.
                    spike_e[t + 1, e] = e + 1
                    spt_e[t + 1, e] = step_times[t + 1] + self.comp_time

                    # Reset Neuron Potential and Refactory Period.
                    v_e[t + 1, e] = V_RESET
                    self.ref_e[0, e] = REFRACTORY

                    # Update Postsynaptic Connections.
                    for i in range(self.num_i):  # E to I
                        if self.s_key_ie[e, i] != 0:
                            syn_idx = int(self.s_key_ie[e, i]) - 1
                            x_ie[t + 1, i] = (  # + W(t) * δ(t - tpre + tdelay)
                                x_ie[t, i] + w_ie[t, syn_idx]
                            )
                            # plasticity update - "on_pre"
                            apre[t + 1, syn_idx] = (  # + A0 * δ(t - tpre)
                                apre[t, syn_idx] + dApre_0
                            )
                            w_ie[t + 1, syn_idx] = (  # equation (9) in paper
                                w_ie[t, syn_idx]
                                + eta * a_LTD * apost[t, syn_idx]
                            )
                            # max synaptic weight check
                            if (self.j_i * w_ie[t + 1, syn_idx]) < MAX_WEIGHT_IE:
                                w_ie[t + 1, syn_idx] = MAX_WEIGHT_IE / self.j_i
                elif self.ref_e[0, e] >= 0:  # in refractory period
                    v_e[t + 1, e] = V_RESET
                elif v_e[t + 1, e] < V_REST:
                    v_e[t + 1, e] = V_REST

            for i in range(self.num_i):  # inhibitory
                if v_i[t, i] < V_THRESHOLD <= v_i[t + 1, i]:
                    # Record Spike.
                    spike_i[t + 1, i] = i + 1 + self.num_e
                    spike_I_time[t + 1, i] = step_times[t + 1]

                    # Reset Neuron Potential and Refactory Period.
                    v_i[t + 1, i] = V_RESET
                    self.ref_i[0, i] = REFRACTORY

                    # Update Postsynaptic Connections.
                    for e in range(self.num_e):  # I to E
                        if self.s_key_ei[i, e] != 0:
                            syn_idx = int(self.s_key_ei[i, e]) - 1
                            x_ei[t + 1, e] = x_ei[t, e] - WEI
                        # plasticity update - "on_post"
                        if self.s_key_ie[e, i] != 0:
                            syn_idx = int(self.s_key_ie[e, i]) - 1
                            apost[t + 1, syn_idx] = apost[t, syn_idx] + dApost_0
                            w_ie[t + 1, syn_idx] = (  # equation (8) in paper
                                w_ie[t, syn_idx]
                                + eta * a_LTP * apre[t, syn_idx]
                            )
                            # max synaptic weight check
                            if (self.j_i * w_ie[t + 1, syn_idx]) > MAX_WEIGHT_EI:
                                w_ie[t + 1, syn_idx] = MAX_WEIGHT_EI / self.j_i
                elif self.ref_i[0, i] >= 0:
                    # check if in refractory period
                    v_i[t + 1, i] = V_RESET
                elif v_i[t + 1, i] < V_REST:
                    v_i[t + 1, i] = V_REST

        # Calculate Synchrony.
        # todo: what synchrony measurement is this?
        # todo: different from Kuramoto?
        N = self.num_e  # + self.num_i;
        Vcomb = np.zeros((num_steps + 1, N))
        Vcomb[:, 0:self.num_e] = v_e
        V1 = np.mean(Vcomb, axis=1)

        # variance of average voltage over whole run
        # sigma_v^2 = <(V(t)^2>t -[<V(t)>]^2
        sigma_squ_v = np.mean(V1 ** 2) - (np.mean(V1)) ** 2

        # variance of voltage at each time step
        sigma_vi = np.zeros(N)
        sum_sig = 0
        for i in range(N):
            sigma_vi[i] = (
                np.mean(Vcomb[:, i] ** 2)
                - (np.mean(Vcomb[:, i])) ** 2
            )
            sum_sig = sum_sig + sigma_vi[i]

        syn_squ = sigma_squ_v / (sum_sig / N)
        synchrony = float(np.sqrt(syn_squ))

        assert step_times.shape == (num_steps + 1,)
        assert v_e.shape == (num_steps + 1, self.num_e)
        assert v_i.shape == (num_steps + 1, self.num_i)
        assert s_ei.shape == (num_steps + 1, self.num_e)
        assert s_ie.shape == (num_steps + 1, self.num_i)
        assert x_ei.shape == (num_steps + 1, self.num_e)
        assert x_ie.shape == (num_steps + 1, self.num_i)
        assert apost.shape == (num_steps + 1, self.num_synapses_ie)
        assert apre.shape == (num_steps + 1, self.num_synapses_ie)
        assert w_ie.shape == (num_steps + 1, self.num_synapses_ie)
        assert spike_e.shape == (num_steps + 1, self.num_e)
        assert spike_i.shape == (num_steps + 1, self.num_i)
        assert spt_e.shape == (num_steps + 1, self.num_e)
        assert self.ref_e.shape == (1, self.num_e)
        assert self.ref_i.shape == (1, self.num_i)
        assert self.phif.shape == (1, self.num_e)

        return (
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
            self.ref_e,
            self.ref_i,
            synchrony,
            spt_e,
            self.phif,
        )


def make_synaptic_connections(num_pre, num_post, epsilon):
    """
    Returns a lookup table for synaptic connections and the number of
    connections made.

    A connection is established with a probability defined by `epsilon`.

    The synapse at `lut[pre, post]` represents a synaptic connection from a
    pre-synaptic neuron `pre` to a post-synaptic neuron `post`.
    """

    # synaptic connection lookup table
    syn_lut = np.zeros((num_pre, num_post), dtype=int)
    count = 0
    for i in range(num_pre):
        for j in range(num_post):
            if np.random.rand() <= epsilon:
                count += 1
                syn_lut[i, j] = count

    return syn_lut, count


def pulsatile_input(multi, v_stim, t_stim, x, duration, step_size):
    """
    Function to generate pulsatile input for the Ue and Ui

    multi: relative duration of the anodic and cathodic phases
    v_stim: stimulation voltage
    t_stim: pulse width
    x: duration of the neutral phase
    duration: duration of the simulation
    step: time step
    :return: Ue, Ui
    """

    num_steps = int(duration / step_size)

    Ue = np.zeros(num_steps)
    Ui = np.zeros(num_steps)

    # biphasic pulse shape symmetric = 1, asymmetric > 1
    pulse_shape = 1  # uses multi instead?

    # Ue input
    t = 0  # current time
    for i in range(num_steps):
        t += step_size

        # Anodic (negative) phase.
        if 0 <= t < t_stim:
            Ue[i] = -v_stim / multi

        # Cathodic (positive) phase.
        if t_stim <= t < 2 * t_stim + step_size:
            Ue[i] = v_stim

        # Neutral phase.
        if 2 * t_stim + step_size <= t < (2 + x) * t_stim + step_size:
            Ue[i] = 0

        # Additional anodic phase (if multi > 1).
        if (2 + x) * t_stim + step_size <= t < (2 + x + multi - 1) * t_stim:
            Ue[i] = -v_stim / multi

        # Reset time step.
        if t >= (2 + x + multi - 1) * t_stim - 0.01:
            t = 0
            Ue[i] = 0

    # Ui input
    t = 0
    for i in range(num_steps):
        t += step_size

        # cathodic phase
        if 0 <= t < t_stim:
            Ui[i] = v_stim

        # anodic phase
        if t_stim <= t < (multi + 1) * t_stim + step_size:
            Ui[i] = -v_stim / multi

        # neutral phase
        if (multi + 1) * t_stim + step_size <= t < (multi + 1 + x) * t_stim:
            Ui[i] = 0

        if t >= (multi + 1 + x) * t_stim - 0.01:
            t = 0
            Ui[i] = 0

    return Ue, Ui
