# src/signals.py

#I'll be importing a minimal amount of python libraries to facilitate for the math involved
import numpy as np

def sine_wave(freq, sampling_rate, duration, amplitude=1.0):
    """
    Generates a sine wave.
    """
    t = np.arange(0, duration, 1 / sampling_rate)
    x = amplitude * np.sin(2 * np.pi * freq * t)
    return t, x


def two_tone_signal(f1, f2, sampling_rate, duration):
    """
    Gives the sum of two sine waves.
    """
    t = np.arange(0, duration, 1 / sampling_rate)
    x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    return t, x
