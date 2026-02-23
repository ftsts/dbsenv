"""
Computing Neuronal Synchrony.
Uses the Kuramoto Order Parameter (KOP) to compute synchronization.

Kuramoto Order Parameter, R(t).
Average Phase of Neurons, ψ(t).
R(t) * e^iψ(t) = 1/Ne * sum(e^iφk(t)), k from 1 to Ne.

Phase of Neuron k, φk(t).
φk(t) = ( 2π(t{k, i+1} - t) ) / t{k, i+1} - t{k, i}).
note: t{k, i} is the ith spike time for the kth neuron

A highly synchronous network has R(t) = 1.
An aysynchronous network has R(t) = 0.

High level of synchrony at high synaptic weights.
Low level of synchrony at low synaptic weights.
"""

import math
import ctypes
from pathlib import Path
import numpy as np
from tqdm import trange


def kop(
    sptime: np.ndarray,
    t: np.ndarray,
    step_size: float,
    duration: float,
    num_neurons: int,
    fast: bool = True,
):
    """
    Returns a list of KOP values,
    representing the neuronal synchronization over time.

    Calls C implementation if fast is True.
    """

    assert sptime.flags.c_contiguous, "Array must be contiguous."

    if fast:
        return kop_c(sptime, t, step_size, duration, num_neurons)

    return kop_py(sptime, t, step_size, duration, num_neurons)


def kop_py(sptime, t, step_size, duration, num_neurons):
    """Python Implementation."""

    total_steps = len(t)
    num_steps = int(duration / step_size)

    # Compute phases.
    phi = np.zeros((total_steps, num_neurons))
    for n in trange(num_neurons,
                    desc="Computing Neuronal Synchrony",
                    unit="neuron"):
        second_spike = 0
        for i in range(num_steps - 1):
            phi[i, n] = 2 * np.pi * (t[i] - sptime[i, n])
            if sptime[i + 1, n] != sptime[i, n]:
                if second_spike == 1:
                    delt = sptime[i + 1, n] - sptime[i, n]
                    a = int(math.floor(sptime[i, n] / step_size))
                    b = int(math.floor(sptime[i + 1, n] / step_size))
                    # Ensure indices are within bounds
                    if b >= total_steps:
                        b = total_steps - 1
                    unnorm = phi[a:b + 1, n]
                    if delt != 0:
                        phi[a:b + 1, n] = unnorm / delt
                second_spike = 1

    return np.abs(np.mean(np.exp(1j * phi), axis=1))


def kop_c(sptime, t, step_size, duration, num_neurons):
    """Calls C implementaion."""

    here = Path(__file__)
    build_dir = here.parent.parent.parent.parent / "build"  # hawk tuah
    lib_path = build_dir / "libkuramoto.so"  # path to compiled c function

    if not lib_path.exists():
        print(
            "\n==========\n"
            "WARNING:\n"
            f"Failed to find compiled C function in {lib_path}.\n"
            "You may have forgotten to compile the C code for Kuramoto Synchrony.\n"
            "Using Python implementation instead."
            "\n==========\n"
        )
        return kop_py(sptime, t, step_size, duration, num_neurons)

    # Load shared library.
    lib = ctypes.CDLL(str(lib_path))

    # Define the function signature.
    lib.kuramoto_syn.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # sptime
        ctypes.POINTER(ctypes.c_double),  # t
        ctypes.c_double,  # step_size
        ctypes.c_double,  # duration
        ctypes.c_int,  # N
        ctypes.c_int  # total_steps
    ]
    lib.kuramoto_syn.restype = ctypes.POINTER(ctypes.c_double)

    # Convert Numpy arrays to C style arrays.
    total_steps = int(len(t))
    sptime_ptr = (ctypes.POINTER(ctypes.c_double) * total_steps)(
        *(row.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for row in sptime)
    )
    t_ptr = t.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Call C function.
    result_ptr = lib.kuramoto_syn(
        sptime_ptr,
        t_ptr,
        step_size,
        duration,
        num_neurons,
        total_steps
    )
    re = np.ctypeslib.as_array(result_ptr, shape=(total_steps,))

    return re
