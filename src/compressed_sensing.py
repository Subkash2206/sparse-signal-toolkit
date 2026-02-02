# src/compressed_sensing.py

import numpy as np

class SensingOperator:
    """
    A Linear Operator strictly for the DFT sensing matrix.
    Theta = Phi * Psi
    
    Phi: Sampling operator (selects indices t_samples)
    Psi: Inverse DFT basis (transforms s to time domain)
    
    This avoids storing the M x N matrix, reducing memory from O(MN) to O(M + N).
    """
    def __init__(self, t_samples, N_grid):
        self.t_samples = t_samples
        self.N = N_grid
        self.M = len(t_samples)
        
        # Pre-compute normalization factor for IDFT if needed, 
        # but standard np.fft.ifft includes 1/N. 
        # Our original code defined Theta without 1/N in create_sensing_matrix,
        # but reconstruction used 1/N. 
        # Usually, Matching Pursuit works best with unit-norm columns.
        # Theoretical DFT columns have norm 1 if factor is 1/sqrt(N).
        # We will follow the standard FFT conventions and normalize on the fly if needed.
        
    def matvec(self, s):
        """
        Computes y = Theta * s
        1. Inverse DFT of s (to get full time signal x)
        2. Subsample x at t_samples
        """
        # IDFT: s (freq) -> x (time)
        # Using built-in optimized FFT for O(N log N)
        # Note: np.fft.ifft includes 1/N scaling.
        # To match the previous 'exp(...)' logic which was just the sum, 
        # we might need to multiply by N. 
        # However, for MP it's the structure that matters. 
        # Let's use standard orthonormal DFT for best compressed sensing results.
        # Orthonormal IDFT: s -> x * sqrt(N) ?
        # Let's stick to numpy "ortho" norm if possible, or manual.
        
        # Using "ortho" ensures energy conservation and unit norm basis vectors
        x = np.fft.ifft(s) * self.N # Undo 1/N scaling to match original definition (sum of exponentials)
        
        # Subsample
        y = x[self.t_samples]
        return y
    
    def rmatvec(self, y):
        """
        Computes Theta^H * y
        (Conjugate Transpose operation)
        
        Theta = Phi * Psi => Theta^H = Psi^H * Phi^H
        
        1. Phi^H: Place y values into full N-length zero vector at t_samples
        2. Psi^H: Forward DFT
        """
        # 1. Zero padding / Embedding
        x_padded = np.zeros(self.N, dtype=complex)
        x_padded[self.t_samples] = y
        
        # 2. Forward DFT
        # matches conjugate of (unscaled) IDFT elements
        s_proxy = np.fft.fft(x_padded)
        
        # In the original code, Theta elements were exp(j...), col norm was sqrt(M)?
        # Actually col norm of DFT matrix (unnormalized) is sqrt(N*M) or similar.
        # For MP, we need unit norm columns usually.
        # Let's stick to the exact adjoint of matvec.
        # if matvec = P * F_inv * N, then rmatvec = (F_inv * N)^H * P^T
        # = N * (F_inv)^H * P^T
        # (F_inv)^H is (1/N * F)^H = 1/N * F^H.
        # Wait, DFT matrix W, W_inv = 1/N W^H.
        # So W^H = N * W_inv. Be careful.
        
        # Let's simplify:
        # Theta_ik = exp(j 2pi k t_i / N)
        # (Theta^H y)_k = sum_i conj(Theta_ik) * y_i
        # = sum_i exp(-j 2pi k t_i / N) * y_i
        # This is exactly the k-th component of DFT of the padded vector y!
        # And specifically, numpy fft computes sum x[n] exp(-2j...)...
        # So yes, np.fft.fft(x_padded) is correct.
        
        return s_proxy

def create_sensing_matrix(t_samples, N_grid):
    """
    Returns the Linear Operator object instead of a dense matrix.
    Compatible interface for matching_pursuit if updated.
    """
    return SensingOperator(t_samples, N_grid)

def create_dct_dictionary(indices, N):
    """
    Creates a DCT (Discrete Cosine Transform) sensing matrix.
    This is useful for image compression/reconstruction as images are sparse in DCT domain.
    
    indices: Which rows to keep (sampled pixel locations)
    N: Total signal dimension
    
    Returns: M x N matrix where each column is a DCT basis vector
    """
    from scipy.fftpack import dct
    
    # Create full DCT dictionary (N x N)
    # Each column is a DCT basis vector
    Psi = np.zeros((N, N))
    for k in range(N):
        basis = np.zeros(N)
        basis[k] = 1
        # Inverse DCT to get the basis vector in signal domain
        Psi[:, k] = dct(basis, type=2, norm='ortho')
    
    # Subsample to get sensing matrix (M x N)
    return Psi[indices, :]


def matching_pursuit(y, operator, max_iterations=20, tolerance=1e-6):
    """
    Performs Matching Pursuit (MP) using the Linear Operator.
    
    y: Observed samples (M,)
    operator: SensingOperator instance
    """
    # operator handles N implicitly
    N = operator.N
    residual = y.copy()
    s_hat = np.zeros(N, dtype=complex)
    
    # We need column norms of Theta for the projection step
    # For DFT basis (unnormalized columns), norm is sqrt(M) since |exp| = 1 and we sum M entries.
    col_norm_sq = operator.M
    # If we wanted unit norm columns, we'd divide everything.
    # MP usually needs normalized atoms.
    
    for it in range(max_iterations):
        # 1. Compute correlations: <residual, column_k>
        # This is exactly operator.H * residual
        projections = operator.rmatvec(residual)
        
        # 2. Find best match
        k_best = np.argmax(np.abs(projections))
        
        if np.abs(projections[k_best]) < tolerance:
            break
            
        # 3. Update coeff
        # We project residual onto u_k (column k)
        # scale = <u_k, residual> / <u_k, u_k>
        # <u_k, residual> is conj(<residual, u_k>) = conj(projections[k_best])
        # <u_k, u_k> is col_norm_sq
        
        scale = np.conjugate(projections[k_best]) / col_norm_sq
        
        # Update s_hat (accumulate)
        s_hat[k_best] += scale
        
        # 4. Update residual
        # residual = residual - scale * u_k
        # We know u_k is the k-th column of Theta
        # We can generate it by applying matvec to a one-hot vector (delta_k)
        # OR more efficiently, just compute the contribution directly:
        # u_k has elements exp(j... t_val ...).
        # This matches the 'create_sensing_matrix' logic:
        # Theta[i, k] = exp(j * 2 * pi * k * t_val / N)
        
        # Efficient update without full matvec:
        term = scale * np.exp(1j * 2 * np.pi * k_best * operator.t_samples / operator.N)
        residual = residual - term
        
    return s_hat

def matching_pursuit_matrix(y, Theta, max_iterations=20, tolerance=1e-6):
    """
    Performs Matching Pursuit (MP) using a dense matrix.
    
    y: Observed samples (M,)
    Theta: Sensing matrix (M x N)
    max_iterations: Maximum number of iterations
    tolerance: Convergence threshold
    
    Returns: Sparse coefficient vector s_hat (N,)
    """
    M, N = Theta.shape
    residual = y.copy().astype(complex)
    s_hat = np.zeros(N, dtype=complex)
    
    # Pre-compute column norms for efficiency
    col_norms_sq = np.sum(np.abs(Theta)**2, axis=0)
    
    for it in range(max_iterations):
        # 1. Compute correlations with all columns
        # projections[k] = <residual, Theta[:, k]>
        projections = np.dot(np.conjugate(Theta.T), residual)
        
        # 2. Find best match
        k_best = np.argmax(np.abs(projections))
        
        if np.abs(projections[k_best]) < tolerance:
            break
        
        # 3. Update coefficient
        # scale = <column_k, residual> / <column_k, column_k>
        scale = projections[k_best] / col_norms_sq[k_best]
        s_hat[k_best] += scale
        
        # 4. Update residual
        residual = residual - scale * Theta[:, k_best]
    
    return s_hat


def reconstruction_from_sparse(s_hat, N):
    """
    Reconstructs full signal using IFFT.
    Complexity: O(N log N)
    """
    # Previously: sum s[k] * exp(...)
    # This is exactly (N * IFFT(s))
    return np.real(np.fft.ifft(s_hat) * N)
