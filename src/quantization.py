# src/quantization.py

import numpy as np

def quantize(x, num_bits, input_range):
    """
    Simulates an ADC by quantizing the input signal to a specific bit depth.
    
    Parameters:
    x (array-like): Input signal.
    num_bits (int): Bit depth (e.g., 8, 16, 24).
    input_range (tuple): (min_val, max_val) of the ADC input.
    
    Returns:
    np.ndarray: Quantized signal (discrete float values).
    """
    x = np.asarray(x)
    min_val, max_val = input_range
    
    # number of discrete levels
    levels = 2 ** num_bits
    
    # Clip input to range
    x_clipped = np.clip(x, min_val, max_val)
    
    # Normalize to [0, 1]
    x_norm = (x_clipped - min_val) / (max_val - min_val)
    
    # Quantize to integer levels [0, levels-1]
    # We use floor(x * levels) usually, or round.
    # Standard ADC behavior: map [0,1] to [0, 2^B - 1]
    # Note: 2^B levels means we assume the interval is divided into 2^B steps
    # Usually: x_quant_int = round(x_norm * (levels - 1))
    x_quant_int = np.round(x_norm * (levels - 1))
    
    # Map back to float range
    x_quant = x_quant_int / (levels - 1) * (max_val - min_val) + min_val
    
    return x_quant


def calculate_sqnr(x_original, x_quantized):
    """
    Calculates the Signal-to-Quantization-Noise Ratio (SQNR) in dB.
    
    SQNR = 10 * log10(Power_signal / Power_noise)
    """
    x_original = np.asarray(x_original)
    x_quantized = np.asarray(x_quantized)
    
    noise = x_original - x_quantized
    
    p_signal = np.mean(x_original ** 2)
    p_noise = np.mean(noise ** 2)
    
    if p_noise == 0:
        return float('inf')
        
    sqnr = 10 * np.log10(p_signal / p_noise)
    
    return sqnr
