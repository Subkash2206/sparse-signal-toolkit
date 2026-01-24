# Sampling, Aliasing, and Spectral Effects in Discrete Signal Processing

This repository is an experimental study of fundamental concepts in **digital signal processing (DSP)**.  
The goal of the project is to understand — through direct implementation and controlled experiments — how sampling, Fourier analysis, and finite observation affect what we can and cannot infer about a signal.

Rather than relying on library abstractions, the core operations (DFT, reconstruction, windowing) are implemented directly from their mathematical definitions. The emphasis is on correctness, interpretability, and understanding limitations, not performance.

---

## What This Project Studies

This project investigates the following questions:

- How does discrete sampling affect the frequency representation of a signal?
- Under what conditions does aliasing occur, and how can it be predicted?
- When is perfect signal reconstruction possible, and when is information permanently lost?
- How do finite observation windows distort spectra (spectral leakage)?
- What does zero-padding actually change in a Fourier spectrum?

Each question is addressed using controlled experiments where the signal content is known in advance.

---

## Project Structure

src/
├── signals.py # Signal generation (ground truth)
├── dft.py # Discrete Fourier Transform (from definition)
├── reconstruction.py # Ideal sinc-based reconstruction
├── windows.py # Window functions (rectangular, Hann, Hamming)
└── experiments.py # All experiments and visualizations


The code is intentionally modular so that each concept can be studied independently.

---

## Signal Generation

Signals are synthesized directly in the time domain using sums of sinusoids.  
For most experiments, a two-tone signal is used:

x(t) = sin(2π · 50t) + 0.5 · sin(2π · 120t)


Because the signal is constructed explicitly, its frequency content is known beforehand. This makes it possible to distinguish genuine DSP effects from implementation errors.

---

## Discrete Fourier Transform

The Discrete Fourier Transform is implemented directly from its mathematical definition:

X[k] = Σ x[n] · exp(−j · 2πkn / N)


No FFT libraries are used.

This makes the relationship between time-domain samples, frequency bins, and physical frequencies explicit and transparent.

The frequency axis is constructed as:

f_k = (k · f_s) / N


so that all spectral results are interpretable in Hertz.

---

## Aliasing Experiments

Aliasing is studied by intentionally sampling below the Nyquist rate.

Example setup:
- Sampling rate: 180 Hz
- Nyquist frequency: 90 Hz
- Signal component: 120 Hz

The alias frequency is predicted analytically:

f_alias = |120 − 180| = 60 Hz


The frequency-domain plots confirm this prediction exactly: the original 120 Hz component disappears and reappears at 60 Hz.

This demonstrates that aliasing is not random distortion, but a deterministic consequence of undersampling.

---

## Signal Reconstruction

Reconstruction is performed using ideal sinc interpolation:


x(t) = Σ x[n] · sinc((t − nT_s) / T_s)


Two cases are examined:

### Sampling Above Nyquist

- The reconstructed signal passes exactly through all samples
- Both frequency components are preserved
- Reconstruction is theoretically exact

### Sampling Below Nyquist

- The reconstructed signal is smooth and continuous
- The signal corresponds to the aliased spectrum, not the original
- High-frequency information is permanently lost

This illustrates a key principle:

> Reconstruction cannot fix incorrect sampling.

---

## Spectral Leakage and Windowing

Finite observation is equivalent to multiplying a signal by a window in time.  
This causes **spectral leakage**, where energy spreads into nearby frequencies.

Three window functions are implemented and compared:
- Rectangular
- Hann
- Hamming

Using frequencies that do not align with DFT bins, the experiments show:

- Rectangular windows produce sharp peaks with strong sidelobes
- Hann and Hamming windows reduce leakage at the cost of frequency resolution
- The observed spectrum depends on the window, even when the signal and sampling rate are unchanged

This demonstrates the time–frequency tradeoff inherent in spectral analysis.

---

## Zero-Padding

Zero-padding is included to clarify a common misconception.

The signal is padded with zeros before computing the DFT. The results show that:

- Peak locations do not change
- No new frequency components appear
- The spectrum appears smoother due to denser frequency sampling

Zero-padding interpolates the spectrum but does not increase true frequency resolution or recover information.

---

## Key Takeaways

This project demonstrates that:

- Sampling choices determine what information is preserved
- Fourier analysis faithfully reflects the sampled data, even when that data is misleading
- Aliasing is predictable and irreversible
- Reconstruction works only when sampling conditions are satisfied
- Windowing controls spectral distortion but introduces tradeoffs
- Zero-padding affects visualization, not information content

Together, these experiments illustrate both the power and the limits of discrete signal processing.

---

## Scope and Limitations

This project does **not** aim to:
- optimize performance
- implement FFT algorithms
- process real-world audio files
- introduce machine learning methods

The focus is strictly on foundational DSP concepts and their experimental verification.

---

## Running the Experiments

All experiments can be run using:

```bash
python src/experiments.py


Dependencies:

Python 3.x

NumPy

Matplotlib
