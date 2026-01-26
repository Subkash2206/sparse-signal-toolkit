# src/aliasing.py

import numpy as np
from adaptive_reconstruction import estimate_parameters

def detect_high_freq_energy(X, fs, threshold_ratio=0.05):
    """
    Detects potential aliasing by checking for energy accumulation near the Nyquist frequency.
    In many natural signals (which are low-pass), high energy near fs/2 suggests spectral crowding or aliasing.
    """
    N = len(X)
    mag = np.abs(X)[:N//2]
    
    # Define "High Frequency" zone (e.g., top 10% of bandwidth)
    cutoff_idx = int((N // 2) * 0.9)
    
    high_freq_energy = np.sum(mag[cutoff_idx:]**2)
    total_energy = np.sum(mag**2)
    
    ratio = high_freq_energy / (total_energy + 1e-12)
    
    return ratio > threshold_ratio, ratio


def attempt_harmonic_recovery(x, fs):
    """
    Attempts to identify if observed peaks are actually aliased harmonics of a lower fundamental.
    Returns:
        - recovered_peaks: list of {freq, amp} where freq might be > fs/2
        - confidence: string explaining what happened
    """
    # 1. Get current peaks
    peaks = estimate_parameters(x, fs)
    
    if not peaks:
        return [], "No peaks detected"
        
    # Sort by amplitude (descending)
    peaks.sort(key=lambda p: p['amp'], reverse=True)
    
    # Assume the strongest peak is the fundamental (or a low-order harmonic)
    # This is a strong assumption!
    f0 = peaks[0]['freq']
    
    recovered_peaks = []
    recovered_peers_count = 0
    
    # We'll reproduce the peak list but "unfold" frequencies that look like aliases
    # Check each peak
    for p in peaks:
        f_measured = p['freq']
        
        # Is this roughly an integer multiple of f0?
        ratio = f_measured / f0
        nearest_harmonic = round(ratio)
        
        if abs(ratio - nearest_harmonic) < 0.1:
            # It fits the harmonic series normally
            recovered_peaks.append(p)
        else:
            # It does NOT fit. Is it an alias of a higher harmonic?
            # potential true freq = k * f0
            # alias would be abs(k*f0 - m*fs)
            
            # Let's check the next few missing harmonics
            found_alias = False
            for k in range(2, 10):
                f_target = k * f0
                
                # First Nyquist zone alias
                # We assume simple folding: f_alias = |f_target - round(f_target/fs)*fs| is wrong
                # Basic alias mapping: f_alias = min distance to integer multiple of fs
                
                # If sampling rate is fs, 
                # frequencies map to [0, fs/2].
                # f mapped is | f - fs * round(f/fs) | ... actually simple modulo logic is cleaner
                
                # Check simple fold: 
                # Case 1: fs - f_target (if f_target in fs/2 to fs)
                # Case 2: f_target - fs (if f_target > fs)
                # General: distance to nearest N*fs
                
                # Standard relationship: f_meas = min | f_target - Z * fs |
                diff = abs(f_measured - abs(f_target - fs * round(f_target/fs)))
                
                # We need a relaxed tolerance
                if diff < (fs * 0.02): 
                    # MATCH! This peak at f_measured is likely the aliased k-th harmonic
                    new_p = p.copy()
                    new_p['freq'] = f_target # Unfold it!
                    new_p['note'] = f"Unfolded from {f_measured:.1f} Hz"
                    recovered_peaks.append(new_p)
                    found_alias = True
                    recovered_peers_count += 1
                    break
            
            if not found_alias:
                # Can't explain it, leave it as is
                recovered_peaks.append(p)
                
    return recovered_peaks, f"Recovered {recovered_peers_count} aliased harmonics based on f0={f0:.1f}Hz"
