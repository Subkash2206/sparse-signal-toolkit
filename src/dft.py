# src/dft.py

import numpy as np

def dft(x):
    """
    Computes the Discrete Fourier Transform (DFT) of a 1D signal.
    """
    # we take a 1D array of samples (assumed to be finite length and discrete-time)
    x = np.asarray(x, dtype=complex)

    #defines the sample resolution (more samples -> better frequency resolution)
    N = x.shape[0]

    # output array where coefficients are complex numbers
    X = np.zeros(N, dtype=complex)


    # we apply the discrete fourier transform formula directly here
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)

    #frequency domain representation
    return X
