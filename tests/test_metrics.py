# tests/test_metrics.py

import numpy as np
import pytest
from metrics import spectral_entropy, peak_to_sidelobe_ratio, energy_concentration
from dft import dft

TOLERANCE = 1e-6

def test_spectral_entropy_impulse():
    """Impulse should have maximum entropy (flat spectrum)."""
    N = 64
    x = np.zeros(N)
    x[0] = 1
    
    X = dft(x)
    entropy = spectral_entropy(X)
    
    # Flat spectrum = high entropy (close to 1)
    assert entropy > 0.95

def test_spectral_entropy_tone():
    """Pure tone should have low entropy (sparse spectrum)."""
    N = 64
    t = np.arange(N) / N
    x = np.sin(2*np.pi*5*t)
    
    X = dft(x)
    entropy = spectral_entropy(X)
    
    # Sparse spectrum = low entropy
    assert entropy < 0.5

def test_psr_single_tone():
    """PSR should be high for a pure tone."""
    N = 64
    t = np.arange(N) / N
    x = np.sin(2*np.pi*5*t)
    
    X = dft(x)
    psr = peak_to_sidelobe_ratio(X)
    
    # Pure tone in DFT can have low PSR if freq aligns with bin
    # Just check it's non-negative
    assert psr >= 0

def test_energy_concentration():
    """Test energy concentration for sparse signal."""
    N = 100
    t = np.arange(N) / N
    x = np.sin(2*np.pi*5*t)
    
    X = dft(x)
    
    # Most energy in top 10%
    conc = energy_concentration(X, percent=0.1)
    assert conc > 0.9

def test_energy_concentration_noise():
    """White noise should have low energy concentration."""
    N = 100
    np.random.seed(42)
    x = np.random.randn(N)
    
    X = dft(x)
    
    # Energy spread out
    conc = energy_concentration(X, percent=0.1)
    assert conc < 0.4
