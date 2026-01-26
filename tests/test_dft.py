import numpy as np
import pytest
from dft import dft

# Threshold for floating point comparisons
TOLERANCE = 1e-10

def test_dft_vs_numpy():
    """ Verify our DFT matches the optimized NumPy FFT implementation. """
    N = 64
    np.random.seed(42)
    # Generate random complex signal
    x = np.random.randn(N) + 1j * np.random.randn(N)
    
    # Calculate both
    X_our = dft(x)
    X_numpy = np.fft.fft(x)
    
    # assert_allclose is better for floating point array comparisons
    np.testing.assert_allclose(X_our, X_numpy, atol=TOLERANCE)

def test_impulse_response():
    """ 
    DFT of a delta function [1, 0, 0, ...] should be [1, 1, 1, ...] 
    (All frequencies present with equal magnitude and 0 phase)
    """
    N = 16
    x = np.zeros(N)
    x[0] = 1
    
    X = dft(x)
    expected = np.ones(N, dtype=complex)
    
    np.testing.assert_allclose(X, expected, atol=TOLERANCE)

def test_linearity():
    """ DFT(a + b) should equal DFT(a) + DFT(b) """
    N = 32
    a = np.random.randn(N)
    b = np.random.randn(N)
    
    X_sum = dft(a + b)
    X_separate = dft(a) + dft(b)
    
    np.testing.assert_allclose(X_sum, X_separate, atol=TOLERANCE)

def test_parseval():
    """ 
    Parseval's Theorem: Sum of |x[n]|^2 = (1/N) * Sum of |X[k]|^2 
    Total energy in time domain equals total energy in frequency domain (scaled by N).
    """
    N = 32
    x = np.random.randn(N)
    
    energy_time = np.sum(np.abs(x)**2)
    
    X = dft(x)
    energy_freq = np.sum(np.abs(X)**2) / N
    
    assert abs(energy_time - energy_freq) < TOLERANCE
