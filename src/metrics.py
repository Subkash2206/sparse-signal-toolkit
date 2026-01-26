# src/metrics.py

import numpy as np

def spectral_entropy(X):
    """
    Computes the spectral entropy of a frequency domain signal.
    Low entropy indicates a sparse, "peaky" spectrum (good for interpretation).
    High entropy indicates a flat or noisy spectrum (whitenoise-like).
    """
    # Compute power spectrum
    P = np.abs(X)**2
    
    # Normalize to treat it like a probability distribution (sum = 1)
    # We add a small epsilon to avoid division by zero
    P_norm = P / (np.sum(P) + 1e-12)
    
    # Calculate entropy: H = -sum(p * log(p))
    # We limit values to avoid log(0)
    entropy = -np.sum(P_norm * np.log(P_norm + 1e-12))
    
    # Normalize by log(N) so the value is between 0 and 1
    entropy /= np.log(len(X))
    
    return entropy


def peak_to_sidelobe_ratio(X):
    """
    Computes the ratio between the main lobe peak and the highest sidelobe.
    Returns value in decibels (dB).
    Higher is better for distinct feature resolution.
    """
    # Working with magnitude
    mag = np.abs(X)
    
    # Sort magnitude values
    sorted_indices = np.argsort(mag)[::-1]
    
    peak_val = mag[sorted_indices[0]]
    
    # We need to find the "second highest" that isn't just the immediate neighbor 
    # of the main peak (which is part of the same lobe).
    # We'll search for the next local maximum that is separated by a few bins.
    
    # Simple heuristic: Look for the max value outside the main peak's immediate neighborhood.
    # We assume the main lobe width is roughly 3-5 bins depending on window.
    # We'll mask out the peak and its neighbors.
    
    mask_width = 5
    peak_idx = sorted_indices[0]
    
    # Create a mask of valid sidelobes
    mask = np.ones_like(mag, dtype=bool)
    
    # Masking the main lobe region
    start_mask = max(0, peak_idx - mask_width)
    end_mask = min(len(mag), peak_idx + mask_width + 1)
    mask[start_mask:end_mask] = False
    
    if not np.any(mask):
        # If everything is masked (e.g. signal is super short), return 0 dB
        return 0.0
        
    sidelobe_val = np.max(mag[mask])
    
    if sidelobe_val == 0:
        return 100.0 # Effectively infinite
        
    ratio = peak_val / sidelobe_val
    return 20 * np.log10(ratio)


def energy_concentration(X, percent=0.05):
    """
    Measures how much of the total spectral energy is contained 
    within the top 'percent' of bins.
    """
    # Power spectrum
    P = np.abs(X)**2
    total_energy = np.sum(P)
    
    if total_energy == 0:
        return 0.0
        
    # Sort energy contributions
    P_sorted = np.sort(P)[::-1]
    
    # Take top K bins
    k = int(max(1, len(X) * percent))
    top_energy = np.sum(P_sorted[:k])
    
    return top_energy / total_energy
