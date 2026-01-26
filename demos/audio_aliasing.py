# demos/audio_aliasing.py

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import matplotlib.pyplot as plt
import os

def generate_chirp(duration, f_start, f_end, fs):
    """Generates a linear chirp signal."""
    t = np.linspace(0, duration, int(fs * duration))
    # Linear chirp: f(t) = f_start + (f_end - f_start) * t / duration
    # Phase phi(t) = integral(f(t)) = f_start * t + 0.5 * k * t^2
    k = (f_end - f_start) / duration
    phi = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    x = 0.5 * np.sin(phi) # Amplitude 0.5
    return t, x

def main():
    print("Running Audio Aliasing Demo...")
    
    # Ensure directories exist
    os.makedirs("demos/audio", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # 1. Generate High-Res Chirp (Standard Audio Rate)
    fs_high = 44100
    duration = 5.0
    f_start = 0
    f_end = 8000 # Sweep up to 8 kHz
    
    t_high, x_high = generate_chirp(duration, f_start, f_end, fs_high)
    
    wav_path_high = "demos/audio/original_chirp.wav"
    # Convert to 16-bit PCM
    wav.write(wav_path_high, fs_high, (x_high * 32767).astype(np.int16))
    print(f"Saved {wav_path_high} (fs={fs_high} Hz)")
    
    # 2. Simulate Undersampling (Aliasing)
    # Target fs = 8000 Hz. Nyquist = 4000 Hz.
    # The chirp goes up to 8000 Hz, so everything above 4000 Hz should alias back down.
    fs_low = 8000
    
    # Simple decimation (pick every Nth sample)
    # Ratio ≈ 5.5. We'll just resample using valid indices closest to the timeline
    step = fs_high / fs_low
    indices = np.arange(0, len(x_high), step).astype(int)
    x_low = x_high[indices]
    
    wav_path_low = "demos/audio/aliased_chirp.wav"
    wav.write(wav_path_low, fs_low, (x_low * 32767).astype(np.int16))
    print(f"Saved {wav_path_low} (fs={fs_low} Hz)")
    
    # 3. Analyze Spectrograms
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Original
    f, t, Sxx = scipy.signal.spectrogram(x_high, fs_high)
    ax1.pcolormesh(t, f, np.log10(Sxx + 1e-10), shading='gouraud')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title(f'Original Spectrogram (fs={fs_high}) - Clean Sweep')
    ax1.set_ylim(0, 10000)
    
    # Aliased
    f, t, Sxx = scipy.signal.spectrogram(x_low, fs_low)
    ax2.pcolormesh(t, f, np.log10(Sxx + 1e-10), shading='gouraud')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (sec)')
    ax2.set_title(f'Aliased Spectrogram (fs={fs_low}) - The "Bounce"')
    ax2.set_ylim(0, 4000) # Only up to new Nyquist
    
    # Add text annotation explaning the bounce
    ax2.text(2.5, 3000, "Reflected Alias!", color='white', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("plots/11_audio_spectrogram.png")
    print("Spectrogram saved to plots/11_audio_spectrogram.png")

if __name__ == "__main__":
    main()
