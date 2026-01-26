# src/experiments.py

import numpy as np
# using matplotlib to plot the experimental results
import matplotlib.pyplot as plt
import os

# ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# importing the two tone summation of 2 sine waves
from signals import two_tone_signal
# importing the mathematical implementation of DFT
from dft import dft

from reconstruction import sinc_reconstruct

from windows import rectangular, hann, hamming


# defines sampling frequency and observation window
# NOTE: fs intentionally violates Nyquist for 120 Hz
fs = 180
duration = 1.0

t, x = two_tone_signal(50, 120, fs, duration)

# we convert time domain to frequency domain
X = dft(x)

# this tells us where each Fourier bin lives in Hz
freqs = np.arange(len(X)) * fs / len(X)

# plotting time-domain signal
plt.figure()
plt.plot(t, x)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Two-Tone Signal (Aliasing Case, fs = 180 Hz)")
plt.savefig("plots/01_time_domain_signal.png")
plt.show()

# plotting frequency-domain magnitude spectrum
plt.figure()
plt.stem(freqs, np.abs(X) / len(X))
plt.xlim(0, fs / 2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Aliasing in Frequency Domain (fs = 180 Hz)")
plt.savefig("plots/02_frequency_domain_aliasing.png")
plt.show()




# Reconstruction experiment (Nyquist satisfied)
fs_good = 1000
duration = 0.1  # shorter for visualization

t_s, x_s = two_tone_signal(50, 120, fs_good, duration)

t_cont = np.linspace(0, duration, 5000)
x_recon = sinc_reconstruct(x_s, fs_good, t_cont)

plt.figure()
plt.plot(t_cont, x_recon, label="Reconstructed")
plt.plot(t_s, x_s, 'o', label="Samples")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Perfect Reconstruction (fs = 1000 Hz)")
plt.legend()
plt.savefig("plots/03_perfect_reconstruction.png")
plt.show()




# Reconstruction experiment (Aliasing case)
fs_bad = 180
duration = 0.1

t_s, x_s = two_tone_signal(50, 120, fs_bad, duration)

t_cont = np.linspace(0, duration, 5000)
x_recon = sinc_reconstruct(x_s, fs_bad, t_cont)

plt.figure()
plt.plot(t_cont, x_recon, label="Reconstructed")
plt.plot(t_s, x_s, 'o', label="Samples")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Reconstruction after Aliasing (fs = 180 Hz)")
plt.legend()
plt.savefig("plots/04_aliased_reconstruction.png")
plt.show()



# Spectral leakage experiment
fs = 1000
duration = 1.0

# choose a frequency that does NOT align with DFT bins
t, x = two_tone_signal(55, 120, fs, duration)

N = len(x)

windows = {
    "Rectangular": rectangular(N),
    "Hann": hann(N),
    "Hamming": hamming(N)
}

plt.figure(figsize=(10, 6))

for name, w in windows.items():
    Xw = dft(x * w)
    freqs = np.arange(len(Xw)) * fs / len(Xw)
    plt.plot(freqs, np.abs(Xw) / len(Xw), label=name)

plt.xlim(0, 200)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Spectral Leakage with Different Windows")
plt.legend()
plt.savefig("plots/05_spectral_leakage.png")
plt.show()



# Zero-padding experiment
# This experiment studies how zero-padding affects the visual appearance
# of the frequency spectrum without adding new information.

# Sampling parameters (Nyquist satisfied)
fs = 1000
duration = 1.0

# Generate the original two-tone signal
t, x = two_tone_signal(50, 120, fs, duration)

# Compute the DFT of the original signal
X = dft(x)

# Frequency axis for the original DFT
freqs = np.arange(len(X)) * fs / len(X)

# Zero-padding factor
# Padding increases the number of DFT points but does NOT increase
# the observation duration or true frequency resolution
pad_factor = 4

# Create a zero-padded signal
# The original samples occupy the first part of the array,
# and the remaining samples are filled with zeros
x_padded = np.zeros(pad_factor * len(x))
x_padded[:len(x)] = x

# Compute the DFT of the zero-padded signal
X_pad = dft(x_padded)

# Frequency axis for the zero-padded DFT
# Note that the frequency spacing is finer because the DFT length is larger
freqs_pad = np.arange(len(X_pad)) * fs / len(X_pad)

# Plot comparison between original and zero-padded spectra
plt.figure(figsize=(10, 6))

# Original spectrum (coarser frequency sampling)
plt.plot(freqs, np.abs(X) / len(X), label="Original DFT")

# Zero-padded spectrum (denser frequency sampling)
plt.plot(freqs_pad, np.abs(X_pad) / len(X_pad), label="Zero-padded DFT")

plt.xlim(0, 200)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Effect of Zero-Padding on DFT Spectrum")
plt.legend()
plt.savefig("plots/06_zero_padding.png")
plt.show()



# Compressed Sensing Experiment
# Demonstrating recovery of specific frequencies from random sub-Nyquist samples
from compressed_sensing import create_sensing_matrix, matching_pursuit, reconstruction_from_sparse

# 1. Create a sparse signal (sum of 2 sinusoids)
N = 200 # Total points on the grid
fs = 200 # 1 Hz resolution
t_full = np.arange(N) / fs
# Frequencies: 13 Hz and 35 Hz
x_full = 1.0 * np.cos(2 * np.pi * 13 * t_full) + 0.8 * np.cos(2 * np.pi * 35 * t_full)

# 2. Random Sampling (Sub-Nyquist average rate?)
# We take M samples randomly
M = 40 # Only 20% of the data!
np.random.seed(42)
t_indices = np.sort(np.random.choice(N, M, replace=False))
x_sampled = x_full[t_indices]

# 3. Classic Reconstruction? 
# Impossible with standard uniform assumption tools without interpolation, 
# and extremely prone to aliasing if treated as uniform.

# 4. Compressed Sensing Recovery
Theta = create_sensing_matrix(t_indices, N)
s_recovered = matching_pursuit(x_sampled, Theta, max_iterations=10)
x_cs_recon = reconstruction_from_sparse(s_recovered, N)

plt.figure(figsize=(10, 6))
plt.plot(t_full, x_full, 'k-', alpha=0.3, label="Original (Truth)")
plt.plot(t_full, x_cs_recon, 'g--', label="CS Reconstruction (MP)")
plt.plot(t_full[t_indices], x_sampled, 'ro', label="Random Samples (20%)")
plt.title(f"Compressed Sensing Recovery ({M} samples from {N})")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.savefig("plots/07_compressed_sensing.png")
plt.show()
