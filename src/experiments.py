# src/experiments.py

import numpy as np
#using matplot to plot the experimental results
import matplotlib.pyplot as plt

#importing the two tone summation of 2 sine waves
from signals import two_tone_signal
#importing the mathematical implementation of DFT
from dft import dft

fs = 1000
duration = 1.0

t, x = two_tone_signal(50, 120, fs, duration)

X = dft(x)

freqs = np.arange(len(X)) * fs / len(X)


plt.figure()
plt.plot(t, x)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Two-Tone Signal (50 Hz + 120 Hz)")
plt.show()


plt.figure()
plt.stem(freqs, np.abs(X) / len(X))
plt.xlim(0, 200)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("DFT Magnitude Spectrum")
plt.show()
