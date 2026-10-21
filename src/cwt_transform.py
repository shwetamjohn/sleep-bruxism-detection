"""
Continuous Wavelet Transform (CWT) Module for Sleep Bruxism Detection
Converts 1D preprocessed EMG/EEG signals into 2D time-frequency scalogram images,
following the methodology established in Syed Anna's sleep analysis research.
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image
import io

def generate_cwt_scalogram(signal_epoch, fs=512, total_scales=64, wavelet='cmor1.5-1.0'):
    """
    Computes the Continuous Wavelet Transform (CWT) on a 1D signal epoch.
    
    Parameters:
        signal_epoch (np.ndarray): 1D array of the preprocessed signal epoch.
        fs (int): Sampling frequency in Hz.
        total_scales (int): Number of frequency scales to decompose into (default: 64).
        wavelet (str): Wavelet family (Complex Morlet 'cmor1.5-1.0' is ideal for bio-signal bursts).
        
    Returns:
        np.ndarray: 2D scalogram power array (scales x time_samples) normalized to [0, 1].
    """
    scales = np.arange(1, total_scales + 1)
    sampling_period = 1.0 / fs
    
    # Compute CWT coefficients
    coefficients, frequencies = pywt.cwt(
        signal_epoch,
        scales,
        wavelet,
        sampling_period=sampling_period
    )
    
    # Scalogram power is the squared magnitude of the complex coefficients
    power = np.abs(coefficients) ** 2
    
    # Normalize power to [0, 1]
    p_min = np.min(power)
    p_max = np.max(power)
    if p_max - p_min > 0:
        power = (power - p_min) / (p_max - p_min)
        
    return power, frequencies

def scalogram_to_image(power_scalogram, img_size=(224, 224), colormap='jet'):
    """
    Converts a 2D scalogram matrix into a standard 3-channel (RGB) PIL Image of specified dimensions.
    
    Parameters:
        power_scalogram (np.ndarray): 2D normalized scalogram matrix.
        img_size (tuple): Target image dimensions (width, height), standard is (224, 224).
        colormap (str): Matplotlib colormap (e.g., 'jet', 'viridis', or 'magma').
        
    Returns:
        PIL.Image: Resized RGB image ready for CNN / LSKNet.
    """
    cmap = plt.get_cmap(colormap)
    colored_data = cmap(power_scalogram)  # Returns RGBA (values in [0, 1])
    
    # Convert RGBA in [0, 1] to RGB in [0, 255]
    rgb_data = (colored_data[:, :, :3] * 255).astype(np.uint8)
    
    image = Image.fromarray(rgb_data)
    if img_size is not None:
        image = image.resize(img_size, Image.Resampling.BILINEAR)
        
    return image

def save_epoch_scalogram(signal_epoch, output_path, fs=512, total_scales=64, img_size=(224, 224)):
    """
    Convenience function: computes CWT from a 1D epoch and saves it directly to disk as a PNG.
    """
    power, _ = generate_cwt_scalogram(signal_epoch, fs=fs, total_scales=total_scales)
    img = scalogram_to_image(power, img_size=img_size)
    img.save(output_path)
    return output_path
