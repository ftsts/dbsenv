import numpy as np


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
