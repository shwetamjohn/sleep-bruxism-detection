"""
Continuous Wavelet Transform (CWT) Module for Sleep Bruxism Detection
Supports both Single-Signal (EMG) and Multi-Signal (EMG + EEG + ECG) Scalogram Image generation.
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image

def compute_cwt_power(signal_epoch, fs=256, total_scales=64, wavelet='cmor1.5-1.0'):
    """
    Computes Continuous Wavelet Transform (CWT) and returns normalized 2D power array.
    """
    scales = np.arange(1, total_scales + 1)
    sampling_period = 1.0 / fs
    
    coefficients, _ = pywt.cwt(
        signal_epoch,
        scales,
        wavelet,
        sampling_period=sampling_period
    )
    power = np.abs(coefficients) ** 2
    
    p_min = np.min(power)
    p_max = np.max(power)
    if p_max - p_min > 0:
        power = (power - p_min) / (p_max - p_min)
    else:
        power = np.zeros_like(power)
        
    return power

def generate_cwt_scalogram(signal_epoch, fs=256, total_scales=64, wavelet='cmor1.5-1.0'):
    """Single-signal CWT scalogram calculation."""
    power = compute_cwt_power(signal_epoch, fs=fs, total_scales=total_scales, wavelet=wavelet)
    return power, None

def scalogram_to_image(power_scalogram, img_size=(224, 224), colormap='jet'):
    """Converts a single 2D power scalogram into a color-mapped 3-channel RGB image."""
    cmap = plt.get_cmap(colormap)
    colored_data = cmap(power_scalogram)  # RGBA in [0, 1]
    rgb_data = (colored_data[:, :, :3] * 255).astype(np.uint8)
    image = Image.fromarray(rgb_data)
    if img_size is not None:
        image = image.resize(img_size, Image.Resampling.BILINEAR)
    return image

def generate_multisignal_scalogram_image(emg_epoch, eeg_epoch=None, ecg_epoch=None, fs=256, total_scales=64, img_size=(224, 224)):
    """
    Generates a 3-channel Multimodal Scalogram Image:
      - Channel R (Red):   Chin EMG CWT Scalogram (Jaw clench / muscle activity)
      - Channel G (Green): Central EEG CWT Scalogram (Brain micro-arousal)
      - Channel B (Blue):  ECG CWT Scalogram (Autonomic heart rate surge)
      
    If EEG or ECG is not available, falls back to single-signal color-mapping.
    """
    # 1. Compute EMG CWT (Red Channel)
    emg_power = compute_cwt_power(emg_epoch, fs=fs, total_scales=total_scales)
    
    if eeg_epoch is None and ecg_epoch is None:
        # Fallback to single-signal colormap
        return scalogram_to_image(emg_power, img_size=img_size, colormap='jet')
        
    # 2. Compute EEG CWT (Green Channel)
    if eeg_epoch is not None:
        eeg_power = compute_cwt_power(eeg_epoch, fs=fs, total_scales=total_scales)
    else:
        eeg_power = np.zeros_like(emg_power)
        
    # 3. Compute ECG CWT (Blue Channel)
    if ecg_epoch is not None:
        ecg_power = compute_cwt_power(ecg_epoch, fs=fs, total_scales=total_scales)
    else:
        ecg_power = np.zeros_like(emg_power)
        
    # Stack into (H, W, 3) image
    r = (emg_power * 255).astype(np.uint8)
    g = (eeg_power * 255).astype(np.uint8)
    b = (ecg_power * 255).astype(np.uint8)
    
    rgb_data = np.stack([r, g, b], axis=-1)
    image = Image.fromarray(rgb_data)
    if img_size is not None:
        image = image.resize(img_size, Image.Resampling.BILINEAR)
        
    return image
