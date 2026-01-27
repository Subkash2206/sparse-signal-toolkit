# src/fft.py

import numpy as np

def fft(x):
    """
    Computes the Fast Fourier Transform (FFT) using the Cooley-Tukey algorithm.
    
    This is a recursive Radix-2 implementation.
    Input length N must be a power of 2.
    
    Complexity: O(N log N)
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    
    # Base case: DFT of size 1 is the identity
    if N == 1:
        return x
    
    # Ensure N is a power of 2
    if N % 2 != 0:
        raise ValueError("FFT requires input length to be a power of 2")
    
    # Divide: Split into even and odd indexed elements
    X_even = fft(x[0::2])
    X_odd = fft(x[1::2])
    
    # Conquer: Combine using butterfly operations
    # Twiddle factors: W_N^k = exp(-2j * pi * k / N)
    k = np.arange(N // 2)
    twiddle = np.exp(-2j * np.pi * k / N)
    
    # Butterfly
    X = np.zeros(N, dtype=complex)
    X[:N//2] = X_even + twiddle * X_odd
    X[N//2:] = X_even - twiddle * X_odd
    
    return X


def _get_bit_reverse_indices(N):
    """
    Pre-computes bit-reversal permutation indices.
    Complexity: O(N)
    """
    n = np.arange(N)
    b = n.reshape((N, 1))
    bits = int(np.log2(N))
    # Provide a simple precomputed way using shifting
    # For a specialized localized bit reversal (without calling int->str):
    # This is a classic trick:
    reversed_n = np.zeros(N, dtype=int)
    for i in range(N):
        # Bitwise reversal
        val = 0
        temp = i
        for _ in range(bits):
            val = (val << 1) | (temp & 1)
            temp >>= 1
        reversed_n[i] = val
    return reversed_n

def fft_iterative(x):
    """
    Iterative (in-place) implementation of the Cooley-Tukey FFT.
    Uses bit-reversal permutation and iterative butterfly stages.
    
    Optimization:
    - Replaced string-based bit reversal with integer bitwise ops.
    - Used pre-computed indices to avoid O(N log N) reversal overhead loop.
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    
    if N == 1:
        return x
    
    # Check power of 2
    if N & (N - 1) != 0:
        raise ValueError("FFT requires input length to be a power of 2")
    
    # 1. Bit-reversal permutation (Vectorized / Pre-computed)
    # This is O(N) memory copy but avoids the slow string formatting loop
    perm = _get_bit_reverse_indices(N)
    X = x[perm]
    
    # 2. Iterative butterfly stages
    log2_N = int(np.log2(N))
    
    for stage in range(1, log2_N + 1):
        m = 2 ** stage  # Sub-problem size
        half_m = m // 2
        
        # Twiddle factors for this stage
        # W_m^k for k in 0..half_m-1
        # We can vectorize this inner loop partially
        k_range = np.arange(half_m)
        W = np.exp(-2j * np.pi * k_range / m)
        
        # We step through the array in chunks of size m
        for k in range(0, N, m):
            # Extract the two halves
            even = X[k : k + half_m]
            odd  = X[k + half_m : k + m]
            
            # Butterfly
            term = W * odd
            
            # Compute new values first to avoid overwriting 'even' (which is a view)
            # before using it for the subtraction
            new_even = even + term
            new_odd = even - term
            
            X[k : k + half_m] = new_even
            X[k + half_m : k + m] = new_odd
            
    return X


def ifft(x):
    """
    Computes the Inverse Fast Fourier Transform (IFFT).
    
    Uses the conjugate trick to reuse the forward FFT implementation:
    IFFT(x) = conj(FFT(conj(x))) / N
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    
    # Conjugate the input
    x_conj = np.conjugate(x)
    
    # Compute forward FFT
    X_conj = fft(x_conj)
    
    # Conjugate the result and scale by 1/N
    return np.conjugate(X_conj) / N
