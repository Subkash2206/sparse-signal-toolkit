# src/adaptive_reconstruction.py

import numpy as np
from dft import dft

def estimate_parameters(x, fs, threshold_ratio=0.1):
    """
    Analyzes the signal x (sampled at fs) to extract dominant spectral components.
    Returns a list of oscillators defined by frequency, amplitude, and phase.
    
    threshold_ratio: Filter out peaks weaker than this fraction of the max peak.
    """
    # 1. Compute Spectrum
    # We use a Hann window to minimize spectral leakage for better parameter estimation
    # but we must compensate for the window's amplitude attenuation (coherent gain = 0.5)
    N = len(x)
    w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
    x_windowed = x * w
    
    X = dft(x_windowed)
    
    # 2. Find Peaks
    magnitude = np.abs(X)
    
    # Only look at positive half of spectrum
    half_N = N // 2
    mag_half = magnitude[:half_N]
    
    # Simple peak detection
    # We ignore the DC component (index 0) and Nyquist (index half_N) for simplicity
    peaks = []
    
    max_val = np.max(mag_half)
    threshold = max_val * threshold_ratio
    
    # Iterating through bins to find local maxima
    for k in range(1, half_N - 1):
        if mag_half[k] > threshold:
            if mag_half[k] > mag_half[k-1] and mag_half[k] > mag_half[k+1]:
                # We found a peak at index k
                
                # Coarse Frequency
                freq_bin = k
                
                # Spectral Refinement (Parabolic Interpolation)
                # This helps estimate the "true" occurring frequency between bins
                alpha = mag_half[k-1]
                beta = mag_half[k]
                gamma = mag_half[k+1]
                
                p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
                k_star = freq_bin + p
                
                est_freq = k_star * fs / N
                
                # Amplitude and Phase Estimation
                # We approximate amplitude from the peak bin (compensating for window)
                # Hann window coherent gain is 0.5, so we divide by 0.5
                est_amp = (beta / (N / 2)) / 0.5 
                
                # Phase is trickier with windowing/interpolation, 
                # so we take it directly from the peak bin
                est_phase = np.angle(X[k])
                
                peaks.append({
                    "freq": est_freq,
                    "amp": est_amp,
                    "phase": est_phase
                })
                
    return peaks


def oscillator_bank_reconstruct(params, t_continuous):
    """
    Reconstructs the signal as a sum of sinusoids based on estimated parameters.
    This implies a strong prior: the signal is composed of discrete tones.
    """
    x_recon = np.zeros_like(t_continuous)
    
    for p in params:
        f = p["freq"]
        A = p["amp"]
        phi = p["phase"]
        
        # Add component: A * cos(2*pi*f*t + phi)
        # We use cos because phase 'phi' is typically relative to a cosine basis in DFT if using real definition,
        # but our DFT uses exp(-j...), so X[k] = A * exp(j*phi). 
        # Real part of x(t) corresponds to sum of cosines.
        x_recon += A * np.cos(2 * np.pi * f * t_continuous + phi)
        
    return x_recon
