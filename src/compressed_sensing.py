# src/compressed_sensing.py

import numpy as np

def create_sensing_matrix(t_samples, N_grid):
    """
    Creates the sensing matrix (Phi * Psi) for DFT basis.
    
    t_samples: The time indices of the random samples (integers).
    N_grid: The total length of the full signal we want to reconstruct.
    
    Returns:
        Theta: The M x N transformation matrix where M = len(t_samples).
               Rows correspond to the sample times.
               Columns correspond to the DFT frequencies.
    """
    # We are solving y = Theta * s
    # y is our measurement vector (time domain samples)
    # s is our sparse vector (frequency domain coefficients)
    # Theta combines the random sampling (Phi) and the Inverse DFT basis (Psi)
    
    M = len(t_samples)
    Theta = np.zeros((M, N_grid), dtype=complex)
    
    # IDFT Matrix element: exp(j * 2*pi * k * n / N) / N
    # We ignore the 1/N scaling during pursuit and apply it at the end to keep energy comparable
    for i, t_val in enumerate(t_samples):
        for k in range(N_grid):
            Theta[i, k] = np.exp(1j * 2 * np.pi * k * t_val / N_grid)
            
    return Theta

def create_dct_dictionary(t_samples, N_grid):
    """
    Creates the sensing matrix for DCT-II basis.
    Used for real-valued signals like images.
    
    t_samples: Indices of observed samples.
    N_grid: Total signal length.
    
    Returns:
        Theta = Phi * Psi (Measurement * Inverse DCT)
    """
    M = len(t_samples)
    Theta = np.zeros((M, N_grid))
    
    # DCT-II formula for the k-th basis vector
    # The Inverse DCT-II (Orthonormal) matrix element (n, k)
    # x[n] = sum_k s[k] * w[k] * cos(pi * k * (2n + 1) / (2N))
    # where w[k] is normalization factor
    
    # We construct rows corresponding to t_samples
    for i, t_val in enumerate(t_samples):
        for k in range(N_grid):
            # Normalization factor
            if k == 0:
                alpha = np.sqrt(1 / N_grid)
            else:
                alpha = np.sqrt(2 / N_grid)
            
            # Basis element
            Theta[i, k] = alpha * np.cos(np.pi * k * (2 * t_val + 1) / (2 * N_grid))
            
    return Theta

def matching_pursuit(y, Theta, max_iterations=20, tolerance=1e-6):
    """
    Performs Matching Pursuit (MP) to find sparse coefficients 's'.
    
    y: Observed samples (M,)
    Theta: Sensing matrix (M, N)
    
    Returns:
        s_hat: Estimated sparse spectrum (N,)
    """
    M, N = Theta.shape
    residual = y.copy()
    s_hat = np.zeros(N, dtype=complex)
    
    # Normalize columns of Theta for correct projection
    # (Though theoretical DFT columns are equal norm, checking is good practice)
    norms = np.linalg.norm(Theta, axis=0)
    
    for it in range(max_iterations):
        # 1. Project residual onto all dictionary atoms (correlations)
        # We want <residual, column_k>
        projections = np.dot(Theta.conj().T, residual)
        
        # 2. Find the atom with the highest correlation
        k_best = np.argmax(np.abs(projections))
        
        if np.abs(projections[k_best]) < tolerance:
            break
            
        # 3. Update the coefficient for that atom
        # We project residual onto the BEST column vector: u_k
        u_k = Theta[:, k_best]
        scale = np.dot(u_k.conj().T, residual) / (np.dot(u_k.conj().T, u_k))
        
        s_hat[k_best] += scale
        
        # 4. Update residual
        residual = residual - scale * u_k
        
    return s_hat

def reconstruction_from_sparse(s_hat, N):
    """
    Reconstructs the full time-domain signal from the estimated sparse spectrum.
    
    s_hat: Sparse spectrum coefficients (N,)
    N: Total length of signal
    """
    # This is effectively an Inverse DFT
    # x[n] = (1/N) * sum( s[k] * exp(j*2*pi*k*n/N) )
    # Note: Our matching pursuit solved for `s` such that y = Theta * s.
    # Our Theta logic didn't include the 1/N factor usually present in IDFT.
    
    # Standard IDFT
    k_indices = np.arange(N)
    n_indices = np.arange(N)
    
    # Create full IDFT matrix
    grid_n, grid_k = np.meshgrid(n_indices, k_indices)
    
    # We simply sum the contributions of the found coefficients
    x_recon = np.zeros(N, dtype=complex)
    
    for k in range(N):
        if np.abs(s_hat[k]) > 1e-10:
            # Contribution of frequency k
            # x_k = s_hat[k] * exp(j * 2*pi * k * n / N)
            term = s_hat[k] * np.exp(1j * 2 * np.pi * k * n_indices / N)
            x_recon += term
            
    # For real signals, imaginary parts should cancel out if spectrum is symmetric
    # We return the real part
    return np.real(x_recon)
