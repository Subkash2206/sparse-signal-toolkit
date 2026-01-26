import numpy as np
import pytest
from fft import fft, fft_iterative

TOLERANCE = 1e-10

def test_fft_recursive_vs_numpy():
    """ Verify recursive FFT matches NumPy FFT. """
    N = 64
    np.random.seed(42)
    x = np.random.randn(N) + 1j * np.random.randn(N)
    
    X_our = fft(x)
    X_numpy = np.fft.fft(x)
    
    np.testing.assert_allclose(X_our, X_numpy, atol=TOLERANCE)

def test_fft_iterative_vs_numpy():
    """ Verify iterative FFT matches NumPy FFT. """
    N = 64
    np.random.seed(42)
    x = np.random.randn(N) + 1j * np.random.randn(N)
    
    X_our = fft_iterative(x)
    X_numpy = np.fft.fft(x)
    
    np.testing.assert_allclose(X_our, X_numpy, atol=TOLERANCE)

def test_fft_impulse():
    """ FFT of delta function should be all ones. """
    N = 16
    x = np.zeros(N)
    x[0] = 1
    
    X = fft(x)
    expected = np.ones(N, dtype=complex)
    
    np.testing.assert_allclose(X, expected, atol=TOLERANCE)
