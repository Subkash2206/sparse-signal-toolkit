# src/monte_carlo.py

import numpy as np
import matplotlib.pyplot as plt
import os
from compressed_sensing import create_sensing_matrix, matching_pursuit, reconstruction_from_sparse

def add_noise(signal, snr_db):
    """
    Adds Gaussian white noise to the signal to achieve the desired SNR (in dB).
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def run_monte_carlo():
    print("Starting Monte Carlo Simulation...")
    print("Evaluating Compressed Sensing Robustness under Noise and Sparsity.")
    
    # Simulation Parameters
    N = 100 # Signal length
    fs = 100
    trials = 20 # Trials per grid point (increase for smoother results)
    
    # Grid
    snr_levels = [10, 20, 30, 40, 50] # dB
    sampling_ratios = [0.1, 0.2, 0.3, 0.4, 0.5] # Fraction of N
    
    # Results container: Averaged MSE
    results_grid = np.zeros((len(sampling_ratios), len(snr_levels)))
    
    for i, ratio in enumerate(sampling_ratios):
        M = int(N * ratio)
        for j, snr in enumerate(snr_levels):
            
            error_accum = 0.0
            
            for t in range(trials):
                # 1. Generate random sparse signal (2 tones)
                # Random frequencies
                f1 = np.random.randint(5, 45)
                f2 = np.random.randint(5, 45)
                while f1 == f2: f2 = np.random.randint(5, 45)
                
                t_full = np.arange(N) / fs
                x_true = np.cos(2*np.pi*f1*t_full) + 0.5*np.cos(2*np.pi*f2*t_full)
                
                # 2. Random Sampling
                indices = np.sort(np.random.choice(N, M, replace=False))
                x_sampled = x_true[indices]
                
                # 3. Add Noise
                x_noisy = add_noise(x_sampled, snr)
                
                # 4. Reconstruct
                Theta = create_sensing_matrix(indices, N)
                # We relax tolerance for noisy signals to avoid overfitting noise
                s_est = matching_pursuit(x_noisy, Theta, max_iterations=5, tolerance=1e-1)
                x_recon = reconstruction_from_sparse(s_est, N)
                
                # 5. Calculate Error (MSE)
                mse = np.mean((x_true - x_recon)**2)
                error_accum += mse
                
            avg_mse = error_accum / trials
            results_grid[i, j] = avg_mse
            print(f"Ratio: {ratio:.1f}, SNR: {snr}dB -> MSE: {avg_mse:.4f}")

    # Plotting
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    # Log scale for better error visualization
    plt.imshow(np.log10(results_grid + 1e-10), cmap='viridis_r', aspect='auto', origin='lower')
    
    plt.colorbar(label="Log10(Mean Squared Error)")
    
    plt.xticks(ticks=np.arange(len(snr_levels)), labels=snr_levels)
    plt.yticks(ticks=np.arange(len(sampling_ratios)), labels=sampling_ratios)
    
    plt.xlabel("Signal-to-Noise Ratio (dB)")
    plt.ylabel("Sampling Ratio (M/N)")
    plt.title(f"CS Robustness Analysis ({trials} trials/point)")
    
    plt.savefig("plots/08_monte_carlo_heatmap.png")
    print("\nSimulation Complete. Heatmap saved to plots/08_monte_carlo_heatmap.png")

if __name__ == "__main__":
    run_monte_carlo()
