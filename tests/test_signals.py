# tests/test_signals.py

import numpy as np
import pytest
from signals import two_tone_signal

TOLERANCE = 1e-10

def test_two_tone_signal_length():
    """Test that two-tone signal has correct length."""
    fs = 1000
    duration = 1.0
    
    t, x = two_tone_signal(50, 120, fs, duration)
    
    expected_length = int(fs * duration)
    assert len(t) == expected_length
    assert len(x) == expected_length

def test_two_tone_signal_frequencies():
    """Test that two-tone signal contains correct frequencies."""
    from dft import dft
    
    fs = 1000
    duration = 1.0
    f1, f2 = 50, 120
    
    t, x = two_tone_signal(f1, f2, fs, duration)
    
    # Compute DFT
    X = dft(x)
    freqs = np.arange(len(X)) * fs / len(X)
    
    # Find peaks
    mag = np.abs(X[:len(X)//2])
    peaks = np.argsort(mag)[-2:]
    
    # Check that peaks are near f1 and f2
    peak_freqs = freqs[peaks]
    
    # Should have peaks near 50 and 120 Hz
    assert any(abs(peak_freqs - f1) < 2)
    assert any(abs(peak_freqs - f2) < 2)

def test_two_tone_signal_time_range():
    """Test that time array has correct range."""
    fs = 500
    duration = 0.5
    
    t, x = two_tone_signal(10, 20, fs, duration)
    
    assert t[0] == 0
    assert abs(t[-1] - (duration - 1/fs)) < TOLERANCE
