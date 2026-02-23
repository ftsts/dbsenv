import numpy as np


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
