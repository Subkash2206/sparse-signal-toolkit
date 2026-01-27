# src/filters.py

import numpy as np
from windows import hamming

def low_pass_filter(fc, fs, num_taps):
    """
    Design a Low Pass Filter using the Windowed Sinc method.
    
    Parameters:
    fc (float): Cutoff frequency in Hz.
    fs (float): Sampling frequency in Hz.
    num_taps (int): Length of the filter kernel (should be odd).
    
    Returns:
    np.ndarray: Filter impulse response h[n].
    """
    if num_taps % 2 == 0:
        raise ValueError("Number of taps must be odd for a Type I FIR filter.")
        
    # Normalized cutoff frequency (Nyquist = 0.5)
    f_norm = fc / fs
    
    # Filter order (M)
    M = (num_taps - 1) // 2
    
    # Generate time indices centered around M
    n = np.arange(num_taps)
    
    # Ideal Sinc: h_ideal[n] = 2 * f_norm * sinc(2 * f_norm * (n - M))
    # Handling the 0/0 case where n == M explicitly or using np.sinc which handles it.
    # Note: np.sinc(x) is sin(pi*x)/(pi*x).
    # We want sin(2*pi*f_norm*(n-M)) / (pi*(n-M))
    # = 2 * f_norm * sin(2*pi*f_norm*(n-M)) / (2*pi*f_norm*(n-M))
    # = 2 * f_norm * np.sinc(2 * f_norm * (n - M))
    
    h = 2 * f_norm * np.sinc(2 * f_norm * (n - M))
    
    # Apply Hamming window
    w = hamming(num_taps)
    h_windowed = h * w
    
    # Normalize gain to 1 at DC (sum of coefficients)
    # DC gain is the sum of coefficients.
    h_normalized = h_windowed / np.sum(h_windowed)
    
    return h_normalized


def convolve(x, h):
    """
    Naive 1D convolution of signal x with kernel h.
    y[n] = sum_{k=0}^{M-1} h[k] * x[n-k]
    
    Parameters:
    x (array-like): Input signal.
    h (array-like): Filter kernel.
    
    Returns:
    np.ndarray: Convolved result (mode='full').
    """
    x = np.asarray(x)
    h = np.asarray(h)
    
    N = len(x)
    M = len(h)
    
    # Result length for 'full' convolution is N + M - 1
    y_len = N + M - 1
    y = np.zeros(y_len)
    
    # Naive nested loop implementation
    for n in range(y_len):
        # Convolution sum
        val = 0.0
        for k in range(M):
            if 0 <= n - k < N:
                val += h[k] * x[n - k]
        y[n] = val
        
    return y
