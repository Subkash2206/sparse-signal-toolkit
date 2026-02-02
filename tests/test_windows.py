# tests/test_windows.py

import numpy as np
import pytest
from windows import rectangular, hann, hamming

TOLERANCE = 1e-10

def test_rectangular_unity():
    """Rectangular window should be all ones."""
    N = 100
    w = rectangular(N)
    assert len(w) == N
    np.testing.assert_allclose(w, np.ones(N), atol=TOLERANCE)

def test_hann_endpoints():
    """Hann window should be zero at endpoints."""
    N = 100
    w = hann(N)
    assert len(w) == N
    assert abs(w[0]) < TOLERANCE
    assert abs(w[-1]) < TOLERANCE

def test_hamming_endpoints():
    """Hamming window should be non-zero at endpoints."""
    N = 100
    w = hamming(N)
    assert len(w) == N
    # Hamming has small non-zero values at endpoints
    assert w[0] > 0
    assert w[-1] > 0

def test_window_symmetry():
    """All windows should be symmetric."""
    N = 101  # Odd length
    
    for window_func in [rectangular, hann, hamming]:
        w = window_func(N)
        # Check symmetry
        np.testing.assert_allclose(w, w[::-1], atol=TOLERANCE)
