# src/adaptive_windows.py

import numpy as np
from dft import dft
from windows import rectangular, hann, hamming
from metrics import spectral_entropy, peak_to_sidelobe_ratio, energy_concentration

class WindowSelector:
    """
    Automatically selects the most appropriate window function for spectral analysis.
    It runs the DFT with multiple window types and selects the one that maximizes
    spectral interpretability metrics (sharp peaks, low leakage).
    """
    
    def __init__(self):
        self.windows = {
            "Rectangular": rectangular,
            "Hann": hann,
            "Hamming": hamming
        }
        
    def evaluate(self, x):
        """
        Applies all available windows to signal x and computes metrics.
        Returns a dictionary of results.
        """
        results = {} # Container for all metrics per window
        
        N = len(x)
        
        for name, window_func in self.windows.items():
            # Apply window
            w = window_func(N)
            x_windowed = x * w
            
            # Compute DFT from first principles
            X = dft(x_windowed)
            
            # Compute metrics
            ent = spectral_entropy(X)
            psr = peak_to_sidelobe_ratio(X)
            conc = energy_concentration(X)
            
            # Composite score:
            # We want LOW entropy and HIGH PSR.
            # We weight them experimentally.
            # PSR is in dB (typically 10-60). Entropy is 0-1.
            # A lower entropy is significantly better.
            score = psr - (ent * 50) 
            
            results[name] = {
                "spectrum": X,
                "entropy": ent,
                "psr_dB": psr,
                "concentration": conc,
                "score": score
            }
            
        return results

    def select(self, x):
        """
        Selects the best window for signal x.
        Returns (best_window_name, metrics_summary)
        """
        results = self.evaluate(x)
        
        # Find window with max score
        best_name = max(results, key=lambda k: results[k]["score"])
        
        return best_name, results
