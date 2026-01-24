# src/dft.py

import numpy as np

def dft(x):
    """
    Computes the Discrete Fourier Transform (DFT) of a 1D signal.
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    X = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)

    return X
