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


    # Implementation:
    # X[k] = sum_n x[n] * exp(-2j * pi * k * n / N)
    # This can be written as a Matrix-Vector multiplication: X = W @ x
    # where W[k, n] = exp(-2j * pi * k * n / N)
    
    k = np.arange(N).reshape((N, 1))  # Column vector
    n = np.arange(N).reshape((1, N))  # Row vector
    
    # Broadcast to create N x N matrix of exponents
    exponent = -2j * np.pi * k * n / N
    
    # Compute Twiddle Matrix
    W = np.exp(exponent)
    
    # Matrix-Vector Multiplication
    X = np.dot(W, x)
    
    return X
