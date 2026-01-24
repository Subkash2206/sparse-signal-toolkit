# src/windows.py

import numpy as np


# Rectangular window
# This corresponds to observing the signal over a finite interval
# and abruptly cutting it off at the boundaries.
# In practice, this is equivalent to using no window at all.
def rectangular(N):
    # Returns an array of ones of length N
    # Multiplying a signal by this window leaves it unchanged
    # but introduces strong spectral leakage in the frequency domain
    return np.ones(N)


# Hann window
# This window smoothly tapers the signal to zero at both ends,
# reducing discontinuities at the boundaries of the observation interval.
def hann(N):
    # Create a sample index array from 0 to N-1
    n = np.arange(N)

    # Hann window definition:
    # w[n] = 0.5 * (1 - cos(2πn / (N - 1)))
    # The smooth cosine taper reduces sidelobes in the frequency domain
    return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))


# Hamming window
# Similar to the Hann window, but designed to further suppress
# sidelobe energy at the cost of a slightly wider main lobe.
def hamming(N):
    # Create a sample index array from 0 to N-1
    n = np.arange(N)

    # Hamming window definition:
    # w[n] = 0.54 - 0.46 * cos(2πn / (N - 1))
    # This window provides better sidelobe attenuation than Hann
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
