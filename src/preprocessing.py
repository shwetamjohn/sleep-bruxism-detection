"""
Signal Preprocessing Module for Sleep Bruxism Detection
Handles filtering (bandpass + notch) and amplitude normalization for EMG and EEG signals.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def butter_bandpass_filter(data, lowcut=10.0, highcut=100.0, fs=512, order=4):
    """
    Apply a zero-phase Butterworth band-pass filter to isolate relevant muscle frequencies
    and remove low-frequency baseline drift (breathing, sweat) and high-frequency noise.
    
    Parameters:
        data (np.ndarray): 1D raw signal.
        lowcut (float): Lower cutoff frequency (Hz). Default is 10.0 Hz.
        highcut (float): Upper cutoff frequency (Hz). Default is 100.0 Hz (or 250 Hz for high fs).
        fs (int): Sampling frequency in Hz.
        order (int): Filter order.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    # Ensure frequencies are strictly inside (0, 1)
    high = min(high, 0.99)
    low = max(low, 0.01)
    
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, freq=50.0, fs=512, Q=30):
    """
    Apply a notch filter to eliminate power-line interference (50 Hz or 60 Hz).
    
    Parameters:
        data (np.ndarray): 1D signal.
        freq (float): Powerline frequency to notch out (default 50.0 Hz).
        fs (int): Sampling frequency in Hz.
        Q (float): Quality factor (higher Q means narrower notch).
    """
    if freq >= (0.5 * fs):
        # Notch frequency is above Nyquist; skip
        return data
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data)

def min_max_normalize(data):
    """
    Normalize 1D or 2D array to the range [0, 1].
    """
    d_min = np.min(data)
    d_max = np.max(data)
    if d_max - d_min == 0:
        return np.zeros_like(data)
    return (data - d_min) / (d_max - d_min + 1e-8)

def z_score_normalize(data):
    """
    Standardize data to zero mean and unit variance.
    """
    std = np.std(data)
    if std == 0:
        return np.zeros_like(data)
    return (data - np.mean(data)) / (std + 1e-8)

def preprocess_signal(raw_signal, fs=512, lowcut=10.0, highcut=100.0, notch_freq=50.0):
    """
    End-to-end preprocessing: Bandpass filter -> Notch filter -> Min-Max scaling.
    """
    # 1. Bandpass filter
    bp_filtered = butter_bandpass_filter(raw_signal, lowcut=lowcut, highcut=highcut, fs=fs)
    # 2. Notch filter for 50Hz hum
    notched = notch_filter(bp_filtered, freq=notch_freq, fs=fs)
    # 3. Scale between 0 and 1
    normalized = min_max_normalize(notched)
    return normalized
