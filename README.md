# Sampling & Aliasing DSP Toolkit

**Build Status:** Passing (GitHub Actions)
**Language:** Python 3.10+
**License:** MIT
**Version:** 1.0.0

## Project Overview

The Sampling & Aliasing DSP Toolkit is a specialized Python library developed to investigate the fundamental mathematical properties of digital signal processing. Unlike standard libraries that obscure implementation details, this project provides transparent, ground-up implementations of critical algorithms to facilitate the study of:

1.  **Spectral Analysis:** Comparing computational complexity between naive Discrete Fourier Transforms ($O(N^2)$) and Fast Fourier Transforms ($O(N \log N)$).
2.  **Sampling Theory:** Visualizing the effects of aliasing, spectral folding, and leakage using configurable window functions.
3.  **Sparse Recovery:** Demonstrating Compressed Sensing principles to reconstruct signals from sub-Nyquist sampling rates using Matching Pursuit.

The codebase includes a rigorous test suite, a continuous integration pipeline, and a benchmarking module to validate numerical precision and runtime performance against `numpy.fft`.

## Features and Modules

### 1. Fourier Transforms (`src/fft.py`, `src/dft.py`)
* **Recursive FFT:** Implements the Cooley-Tukey Radix-2 algorithm using divide-and-conquer recursion. Handles complex inputs and enforces power-of-two length constraints.
* **Iterative FFT:** An optimized in-place implementation of Cooley-Tukey using bit-reversal permutation to minimize recursion overhead and improve memory locality.
* **Naive DFT:** A direct implementation of the summation formula $\sum x[n] e^{-j2\pi kn/N}$. This serves as a baseline for correctness verification and performance degradation analysis.

### 2. Compressed Sensing (`src/compressed_sensing.py`)
* **Sensing Matrix Construction:** Utilities to generate partial Fourier matrices and DCT (Discrete Cosine Transform) dictionaries for sparse bases.
* **Matching Pursuit (MP):** A greedy iterative algorithm that recovers sparse coefficients by successively projecting the residual signal onto the dictionary atoms with the highest correlation.
* **Sparse Reconstruction:** Reconstructs the full time-domain signal from the estimated sparse coefficient vector.

### 3. Signal Reconstruction (`src/reconstruction.py`)
* **Whittaker-Shannon Interpolation:** Implements ideal sinc interpolation to reconstruct continuous-time waveforms from discrete samples, demonstrating perfect reconstruction under Nyquist conditions.

### 4. Spectral Analysis Tools
* **Window Functions (`src/windows.py`):**
    * **Rectangular:** Default window, high spectral leakage.
    * **Hann:** Cosine-based taper to suppress side lobes ($0.5(1 - \cos(2\pi n/N))$).
    * **Hamming:** Optimized coefficients ($0.54 - 0.46\cos(\dots)$) for side-lobe cancellation.
* **Aliasing Detection (`src/aliasing.py`):**
    * **High-Frequency Energy Ratio:** Heuristic function to detect potential aliasing by measuring energy concentration near the Nyquist limit.
    * **Harmonic Recovery:** Experimental logic to identify if observed spectral peaks are aliased versions of lower harmonics.
* **Metrics (`src/metrics.py`):**
    * **Spectral Entropy:** Measures the "peakiness" of the power spectrum.
    * **Peak-to-Sidelobe Ratio (PSR):** Quantifies the dynamic range of the spectral analysis in decibels.
    * **Energy Concentration:** Calculates the percentage of total energy contained within the top percentile of frequency bins.

## Installation and Setup

### Prerequisites
* Python 3.10 or higher
* pip package manager

### Dependencies
The project relies on a minimal set of scientific computing libraries:
* `numpy`: Array manipulation and linear algebra.
* `scipy`: Signal processing utilities (used in demos).
* `matplotlib`: Visualization and plotting.
* `pytest`: Unit testing framework.

### Setup Instructions
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/subkash2206/sampling-aliasing-dsp.git](https://github.com/subkash2206/sampling-aliasing-dsp.git)
    cd sampling-aliasing-dsp
    ```

2.  **Install Dependencies**
    ```bash
    pip install numpy scipy matplotlib pytest
    ```

3.  **Verify Installation**
    Run the test suite to ensure all modules are functioning correctly.
    ```bash
    pytest
    ```

## Usage and Demos

The `demos/` directory contains executable scripts illustrating key concepts.

### 1. Image Inpainting via Compressed Sensing
**File:** `demos/image_inpainting.py`
Demonstrates the recovery of a 2D image from only 50% of its pixels.
* **Method:** Uses Matching Pursuit with a DCT dictionary.
* **Process:** Randomly masks 50% of the pixels, solves for the sparse representation, and reconstructs the full image.
* **Output:** Generates a side-by-side comparison (Original vs. Corrupted vs. Reconstructed) saved to `plots/10_image_inpainting.png`.

### 2. Audio Aliasing Simulation
**File:** `demos/audio_aliasing.py`
Auditory and visual demonstration of spectral folding.
* **Process:** Generates a linear chirp sweeping from 0 Hz to 8000 Hz at 44.1 kHz. Decimates the signal to simulate an 8 kHz sampling rate (Nyquist = 4 kHz).
* **Result:** High frequencies "bounce" off the Nyquist limit and fold back into the audible range.
* **Output:**
    * `demos/audio/original_chirp.wav` (High fidelity)
    * `demos/audio/aliased_chirp.wav` (Aliased artifacts)
    * `plots/11_audio_spectrogram.png` (Visualizes the reflection)

### 3. Sampling Experiments
**File:** `src/experiments.py`
A collection of fundamental experiments:
* **Time Domain:** Visualizing two-tone signals.
* **Aliasing:** Reconstruction errors when $f_{signal} > f_s / 2$.
* **Spectral Leakage:** Comparison of Hann, Hamming, and Rectangular windows on non-integer cycle signals.
* **Zero Padding:** Effect of padding on DFT bin density (interpolation vs. resolution).

## Benchmarking

**File:** `benchmarks.py`

This module performs a rigorous performance analysis of the custom implementations against the highly optimized `numpy.fft` library.

* **Methodology:**
    * Runs algorithms on complex random inputs of varying lengths $N$ (powers of 2 from 16 to 4096).
    * Repeats each trial 3 times to average out system jitter.
    * Separates validation for $O(N^2)$ algorithms (stops at $N=512$) to avoid excessive runtime.
* **Comparisons:**
    * Custom DFT ($O(N^2)$)
    * Custom Recursive FFT ($O(N \log N)$)
    * Custom Iterative FFT ($O(N \log N)$)
    * NumPy FFT (C-optimized $O(N \log N)$)
* **Execution:**
    ```bash
    python benchmarks.py
    ```
    Saves a log-log complexity graph to `plots/09_fft_benchmark.png`.

## Testing Strategy

The project utilizes `pytest` for automated unit testing. Tests are located in the `tests/` directory.

### Test Coverage
* **`tests/test_fft.py`**:
    * `test_fft_recursive_vs_numpy`: Validates recursive FFT output matches NumPy within $1e-10$ tolerance.
    * `test_fft_iterative_vs_numpy`: Validates iterative FFT output accuracy.
    * `test_fft_impulse`: Ensures FFT of a delta function yields a flat spectrum (all ones).
* **`tests/test_dft.py`**:
    * `test_dft_vs_numpy`: Checks naive DFT accuracy against FFT.
    * `test_linearity`: Verifies $\text{DFT}(a + b) = \text{DFT}(a) + \text{DFT}(b)$.
    * `test_parseval`: Confirms conservation of energy between time and frequency domains (Parseval's Theorem).

### Continuous Integration
A GitHub Actions workflow (`.github/workflows/python-app.yml`) automatically triggers on every push and pull request to `main`. It sets up a Python 3.10 environment, installs dependencies, and runs the full pytest suite to prevent regression.

## Project Hierarchy

```text
sampling-aliasing-dsp/
├── src/                          # Core source code
│   ├── adaptive_reconstruction.py # Parameter estimation for oscillator banks
│   ├── adaptive_windows.py       # Logic for selecting optimal window functions
│   ├── aliasing.py               # Aliasing detection and harmonic recovery
│   ├── compressed_sensing.py     # Matching Pursuit and Sensing Matrix
│   ├── dft.py                    # Discrete Fourier Transform implementation
│   ├── fft.py                    # Recursive and Iterative FFT implementations
│   ├── experiments.py            # Sampling and leakage experiments
│   ├── metrics.py                # Spectral entropy and PSR metrics
│   ├── monte_carlo.py            # Robustness simulations for CS
│   ├── reconstruction.py         # Sinc interpolation
│   ├── signals.py                # Signal generation primitives
│   └── windows.py                # Window function definitions
├── demos/                        # End-to-end demonstrations
│   ├── audio/                    # Generated audio artifacts
│   ├── audio_aliasing.py         # Audio downsampling demo
│   └── image_inpainting.py       # Image reconstruction demo
├── tests/                        # Unit tests
│   ├── conftest.py               # Pytest configuration and path setup
│   ├── test_dft.py               # Tests for DFT module
│   └── test_fft.py               # Tests for FFT module
├── plots/                        # Generated output visualizations
├── .github/workflows/            # CI/CD configuration
│   └── python-app.yml
├── .gitignore                    # Git ignore rules
├── benchmarks.py                 # Performance benchmarking script
├── pytest.ini                    # Test runner configuration
└── README.md                     # Project documentation

```


# License
This project is licensed under the MIT License. You are free to use, modify, and distribute this software in compliance with the license terms.
