# tests/test_reconstruction.py

import numpy as np
import pytest
from reconstruction import sinc_reconstruct

TOLERANCE = 1e-6

def test_sinc_reconstruct_at_sample_points():
    """Reconstruction should match original samples at sample points."""
    fs = 100
    duration = 0.1
    t_samples = np.arange(0, duration, 1/fs)
    
    # Create signal
    f = 10
    x_samples = np.sin(2*np.pi*f*t_samples)
    
    # Reconstruct at same points
    x_recon = sinc_reconstruct(x_samples, fs, t_samples)
    
    np.testing.assert_allclose(x_recon, x_samples, atol=TOLERANCE)

def test_sinc_reconstruct_bandlimited():
    """Reconstruction should be accurate for bandlimited signal."""
    fs = 100
    duration = 0.2
    t_samples = np.arange(0, duration, 1/fs)
    
    # Create bandlimited signal (f << fs/2)
    f = 5
    x_samples = np.sin(2*np.pi*f*t_samples)
    
    # Reconstruct at finer grid
    t_fine = np.linspace(0, duration, 500)
    x_recon = sinc_reconstruct(x_samples, fs, t_fine)
    
    # Ground truth at fine grid
    x_true = np.sin(2*np.pi*f*t_fine)
    
    # Should be reasonably close
    np.testing.assert_allclose(x_recon, x_true, atol=0.1)
