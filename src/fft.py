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


def fft_iterative(x):
    """
    Iterative (in-place) implementation of the Cooley-Tukey FFT.
    Uses bit-reversal permutation and iterative butterfly stages.
    
    This is more memory-efficient than the recursive version.
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    
    if N == 1:
        return x
    
    # Check power of 2
    if N & (N - 1) != 0:
        raise ValueError("FFT requires input length to be a power of 2")
    
    # Bit-reversal permutation
    log2_N = int(np.log2(N))
    X = np.zeros(N, dtype=complex)
    for i in range(N):
        # Reverse the bits of index i
        reversed_i = int('{:0{width}b}'.format(i, width=log2_N)[::-1], 2)
        X[reversed_i] = x[i]
    
    # Iterative butterfly stages
    for stage in range(1, log2_N + 1):
        m = 2 ** stage  # Current butterfly size
        half_m = m // 2
        
        # Twiddle factor base for this stage
        w_m = np.exp(-2j * np.pi / m)
        
        for k in range(0, N, m):
            w = 1.0
            for j in range(half_m):
                # Butterfly operation
                t = w * X[k + j + half_m]
                u = X[k + j]
                
                X[k + j] = u + t
                X[k + j + half_m] = u - t
                
                w = w * w_m
    
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
