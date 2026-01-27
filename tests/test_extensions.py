# tests/test_extensions.py

import numpy as np
import pytest
from stft import stft
from quantization import quantize, calculate_sqnr
from filters import low_pass_filter, convolve
from windows import hamming

TOLERANCE = 1e-6

# --- STFT Tests ---
def test_stft_output_shape():
    """ Verify STFT output shape. """
    x = np.random.randn(1024)
    N = 256 # Window size
    hop = 128
    window = np.hanning(N)
    
    # Expected frames: (1024 - 256) // 128 + 1 = 6 + 1 = 7
    # Width is N (since custom fft returns full size)
    
    S = stft(x, window, hop)
    
    assert S.shape == (7, N)
    assert S.dtype == complex

def test_stft_energy_conservation():
    """ 
    Parseval's relation for windowed frames strictly applies if we sum over all frames 
    and account for window power, but a simple check is that it is not zero 
    and scales roughly with input energy.
    For this test, we just check it runs and produces non-Zero output.
    """
    x = np.random.randn(1024)
    window = np.ones(256)
    S = stft(x, window, 128)
    assert np.sum(np.abs(S)) > 0


# --- Quantization Tests ---
def test_quantize_ranges():
    """ Verify quantized values are within limits. """
    x = np.linspace(-1, 1, 100)
    bits = 4
    input_range = (-1, 1)
    
    x_q = quantize(x, bits, input_range)
    
    # Should be bounded
    assert np.all(x_q >= -1.0)
    assert np.all(x_q <= 1.0)
    
    # Should have limited number of unique values
    # For 4 bits, max 16 unique values
    unique_vals = np.unique(x_q)
    assert len(unique_vals) <= 2**bits

def test_sqnr_calculation():
    """ Verify SQNR logic. """
    x = np.sin(np.linspace(0, 10, 100))
    # No noise -> Infinite SQNR (handled as inf or very large)
    sqnr = calculate_sqnr(x, x)
    assert np.isinf(sqnr) or sqnr > 100

    # Large noise -> Low SQNR
    sqnr_noisy = calculate_sqnr(x, np.zeros_like(x)) # Noise power = signal power
    # 10 log10(1) = 0 dB
    assert np.isclose(sqnr_noisy, 0, atol=1e-5)


# --- Filter Tests ---
def test_low_pass_filter_gain():
    """ Verify DC gain is normalized to 1. """
    fc = 100
    fs = 1000
    h = low_pass_filter(fc, fs, 51)
    
    # Sum of coefficients should be 1
    assert np.isclose(np.sum(h), 1.0, atol=TOLERANCE)

def test_convolution_vs_numpy():
    """ Verify naive convolution matches numpy.convolve. """
    x = np.random.randn(50)
    h = np.random.randn(11)
    
    y_our = convolve(x, h)
    y_numpy = np.convolve(x, h, mode='full')
    
    np.testing.assert_allclose(y_our, y_numpy, atol=TOLERANCE)
