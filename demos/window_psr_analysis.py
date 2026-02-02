# demos/window_psr_analysis.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from dft import dft
from windows import rectangular, hann, hamming
from metrics import peak_to_sidelobe_ratio

def main():
    print("Running Window Function PSR Analysis...")
    print("=" * 60)
    
    # Signal parameters
    fs = 1000  # Sampling frequency
    duration = 1.0
    N = int(fs * duration)
    
    # Create a SINGLE-TONE signal with non-integer bin frequency to induce spectral leakage
    # This makes PSR analysis meaningful (we want to measure sidelobe suppression)
    f = 55.7  # Hz (not aligned with DFT bins - causes leakage)
    
    t = np.arange(N) / fs
    x = np.sin(2*np.pi*f*t)
    
    # Window functions to test
    windows = {
        "Rectangular": rectangular(N),
        "Hann": hann(N),
        "Hamming": hamming(N)
    }
    
    psr_results = {}
    spectra = {}
    
    print("\nPeak-to-Sidelobe Ratio (PSR) Analysis:")
    print("-" * 60)
    
    for name, window in windows.items():
        # Apply window and compute DFT
        x_windowed = x * window
        X = dft(x_windowed)
        
        # Only look at positive frequencies (first half)
        X_positive = X[:N//2]
        
        # Calculate PSR - find peak and highest sidelobe
        mag = np.abs(X_positive)
        peak_idx = np.argmax(mag)
        peak_val = mag[peak_idx]
        
        # Mask out main lobe (wider for windowed signals)
        if name == "Rectangular":
            mask_width = 2
        else:
            mask_width = 5  # Windowed signals have wider main lobes
        
        mask = np.ones_like(mag, dtype=bool)
        start_mask = max(0, peak_idx - mask_width)
        end_mask = min(len(mag), peak_idx + mask_width + 1)
        mask[start_mask:end_mask] = False
        
        # Find highest sidelobe
        if np.any(mask):
            sidelobe_val = np.max(mag[mask])
            if sidelobe_val > 0:
                psr = 20 * np.log10(peak_val / sidelobe_val)
            else:
                psr = 100.0
        else:
            psr = 0.0
        
        psr_results[name] = psr
        spectra[name] = X
        
        print(f"{name:12s} Window: PSR = {psr:.2f} dB")
    
    # Calculate improvements
    print("\n" + "=" * 60)
    print("PSR Improvements over Rectangular Window:")
    print("-" * 60)
    
    baseline_psr = psr_results["Rectangular"]
    
    for name, psr in psr_results.items():
        if name != "Rectangular":
            improvement = psr - baseline_psr
            print(f"{name:12s}: +{improvement:.2f} dB improvement")
    
    # Plotting
    os.makedirs("plots", exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    freqs = np.arange(N) * fs / N
    
    for idx, (name, X) in enumerate(spectra.items()):
        ax = axes[idx]
        
        # Plot in dB scale for better visualization
        mag_db = 20 * np.log10(np.abs(X) / N + 1e-10)
        
        ax.plot(freqs, mag_db)
        ax.set_xlim(0, 200)
        ax.set_ylim(-120, 0)
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"{name} Window - PSR: {psr_results[name]:.2f} dB")
        ax.grid(True, alpha=0.3)
        
        if idx == 2:
            ax.set_xlabel("Frequency (Hz)")
    
    plt.tight_layout()
    plt.savefig("plots/12_window_psr_comparison.png", dpi=150)
    print(f"\nPlot saved to plots/12_window_psr_comparison.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
