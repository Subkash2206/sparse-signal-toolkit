# src/experiments.py

import numpy as np
# using matplotlib to plot the experimental results
import matplotlib.pyplot as plt

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
plt.show()

# plotting frequency-domain magnitude spectrum
plt.figure()
plt.stem(freqs, np.abs(X) / len(X))
plt.xlim(0, fs / 2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Aliasing in Frequency Domain (fs = 180 Hz)")
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
plt.show()
