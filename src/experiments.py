# src/experiments.py

#importing matplotlib for experimental results
import matplotlib.pyplot as plt
from signals import sine_wave, two_tone_signal

# Parameters
fs = 1000       # sampling rate in Hz
duration = 1.0  # seconds

# Generate signal
t, x = two_tone_signal(50, 120, fs, duration)

# Plot
plt.figure()
plt.plot(t, x)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Two-Tone Signal (50 Hz + 120 Hz)")
plt.show()
