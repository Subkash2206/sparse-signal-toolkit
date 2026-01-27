# src/stft.py

import numpy as np
from src.fft import fft

def stft(x, window, hop_size):
    """
    Computes the Short-Time Fourier Transform (STFT).
    
    Parameters:
    x (array-like): Input signal.
    window (array-like): Window function array. Length defines the frame size (N).
    hop_size (int): Number of samples to shift the window.
    
    Returns:
    np.ndarray: 2D complex array (Time x Frequency).
                Rows correspond to time frames, columns to frequency bins.
    """
    x = np.asarray(x)
    window = np.asarray(window)
    N = len(window)
    L = len(x)
    
    # Calculate number of frames
    # We ignore the last partial frame if it's smaller than N
    num_frames = (L - N) // hop_size + 1
    
    if num_frames <= 0:
        raise ValueError("Input signal is too short for the given window size.")
        
    stft_matrix = []
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + N
        
        # Extract frame
        frame = x[start:end]
        
        # Apply window
        windowed_frame = frame * window
        
        # Compute FFT
        spectrum = fft(windowed_frame)
        
        stft_matrix.append(spectrum)
        
    return np.array(stft_matrix, dtype=complex)
