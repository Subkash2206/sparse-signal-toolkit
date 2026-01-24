# src/reconstruction.py

import numpy as np

def sinc_reconstruct(x, fs, t_continuous):
    """
    Reconstruct a continuous-time signal from samples using sinc interpolation.
    """
    n = np.arange(len(x))
    Ts = 1 / fs

    x_recon = np.zeros_like(t_continuous, dtype=float)

    for i, t in enumerate(t_continuous):
        x_recon[i] = np.sum(
            x * np.sinc((t - n * Ts) / Ts)
        )

    return x_recon
