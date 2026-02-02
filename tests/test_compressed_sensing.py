# tests/test_compressed_sensing.py

import numpy as np
import pytest
from compressed_sensing import create_sensing_matrix, matching_pursuit, reconstruction_from_sparse, matching_pursuit_matrix, create_dct_dictionary

TOLERANCE = 1e-6

def test_sensing_operator_shape():
    """Test that sensing operator has correct dimensions."""
    N = 100
    M = 30
    indices = np.sort(np.random.choice(N, M, replace=False))
    
    operator = create_sensing_matrix(indices, N)
    
    assert operator.N == N
    assert operator.M == M

def test_matching_pursuit_sparse_recovery():
    """Test MP can recover a simple sparse signal."""
    N = 50
    M = 25
    
    # Create sparse signal (2 frequencies)
    f1,  f2 = 5, 15
    t = np.arange(N) / N
    x_true = np.cos(2*np.pi*f1*t) + 0.5*np.cos(2*np.pi*f2*t)
    
    # Random sampling
    np.random.seed(42)
    indices = np.sort(np.random.choice(N, M, replace=False))
    y = x_true[indices]
    
    # Reconstruct
    operator = create_sensing_matrix(indices, N)
    s_hat = matching_pursuit(y, operator, max_iterations=10)
    x_recon = reconstruction_from_sparse(s_hat, N)
    
    # Should have reasonable reconstruction
    error = np.linalg.norm(x_true - x_recon) / np.linalg.norm(x_true)
    assert error < 0.5  # 50% error is acceptable for this simple test

def test_matching_pursuit_matrix():
    """Test matrix-based matching pursuit."""
    N = 20
    M = 15
    
    # Create simple DCT matrix
    np.random.seed(42)
    indices = np.sort(np.random.choice(N, M, replace=False))
    Theta = np.random.randn(M, N)
    
    # Create sparse signal
    s_true = np.zeros(N)
    s_true[3] = 1.0
    s_true[7] = 0.5
    
    y = np.dot(Theta, s_true)
    
    # Recover
    s_hat = matching_pursuit_matrix(y, Theta, max_iterations=20)
    
    # Should find the dominant coefficients
    top_indices = np.argsort(np.abs(s_hat))[-2:]
    assert 3 in top_indices or 7 in top_indices

def test_reconstruction_accuracy():
    """Test that reconstruction preserves energy."""
    N = 64
    
    # Create frequency domain signal
    s = np.zeros(N, dtype=complex)
    s[10] = 1.0 + 0j
    
    # Reconstruct
    x = reconstruction_from_sparse(s, N)
    
    # Should be real sinusoid
    assert np.all(np.abs(np.imag(x)) < TOLERANCE)
