# benchmarks.py

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dft import dft
from fft import fft, fft_iterative

def benchmark(func, x, repeats=3):
    """Run function multiple times and return average execution time."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(x)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times)

def run_benchmarks():
    print("Running FFT Benchmarks...")
    print("=" * 50)
    
    # Test sizes (powers of 2)
    sizes = [2**n for n in range(4, 13)]  # 16 to 4096
    
    dft_times = []
    fft_rec_times = []
    fft_iter_times = []
    numpy_times = []
    
    for N in sizes:
        print(f"N = {N:5d} ... ", end="", flush=True)
        
        # Generate random signal
        np.random.seed(42)
        x = np.random.randn(N) + 1j * np.random.randn(N)
        
        # Only run DFT for small sizes (it's too slow otherwise)
        if N <= 512:
            t_dft = benchmark(dft, x)
            dft_times.append(t_dft)
        else:
            dft_times.append(np.nan)
        
        # FFT (recursive)
        t_fft_rec = benchmark(fft, x)
        fft_rec_times.append(t_fft_rec)
        
        # FFT (iterative)
        t_fft_iter = benchmark(fft_iterative, x)
        fft_iter_times.append(t_fft_iter)
        
        # NumPy FFT
        t_numpy = benchmark(np.fft.fft, x)
        numpy_times.append(t_numpy)
        
        print(f"DFT: {dft_times[-1] if not np.isnan(dft_times[-1]) else 'SKIP':>10} | "
              f"FFT(rec): {t_fft_rec:.6f}s | "
              f"FFT(iter): {t_fft_iter:.6f}s | "
              f"NumPy: {t_numpy:.6f}s")
    
    # Plotting
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    plt.loglog(sizes, dft_times, 'r-o', label="O(N²) DFT", markersize=8)
    plt.loglog(sizes, fft_rec_times, 'b-s', label="O(N log N) FFT (Recursive)", markersize=6)
    plt.loglog(sizes, fft_iter_times, 'g-^', label="O(N log N) FFT (Iterative)", markersize=6)
    plt.loglog(sizes, numpy_times, 'k--d', label="NumPy FFT (Optimized C)", markersize=6)
    
    plt.xlabel("Signal Length (N)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("FFT Benchmark: Complexity Comparison")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.savefig("plots/09_fft_benchmark.png", dpi=150)
    print("\nBenchmark plot saved to plots/09_fft_benchmark.png")

if __name__ == "__main__":
    run_benchmarks()
