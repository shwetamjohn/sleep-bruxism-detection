"""
Signal Preprocessing Module for Sleep Bruxism Detection
Handles filtering (bandpass + notch) and amplitude normalization for EMG, EEG, and ECG signals.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def butter_bandpass_filter(data, lowcut=10.0, highcut=100.0, fs=256, order=4):
    """
    Apply a zero-phase Butterworth band-pass filter.
    """
    nyquist = 0.5 * fs
    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.999)
    
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, freq=50.0, fs=256, Q=30):
    """
    Apply a notch filter to eliminate power-line interference (50 Hz or 60 Hz).
    """
    if freq >= (0.5 * fs):
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

def preprocess_emg(raw_signal, fs=256):
    """
    Chin EMG: Focuses on muscle contractions (10 Hz - 100 Hz).
    """
    bp = butter_bandpass_filter(raw_signal, lowcut=10.0, highcut=min(100.0, 0.45 * fs), fs=fs)
    notched = notch_filter(bp, freq=50.0, fs=fs)
    return min_max_normalize(notched)

def preprocess_eeg(raw_signal, fs=256):
    """
    Central EEG: Focuses on sleep rhythm & micro-arousal frequencies (0.5 Hz - 35 Hz).
    """
    bp = butter_bandpass_filter(raw_signal, lowcut=0.5, highcut=min(35.0, 0.45 * fs), fs=fs)
    notched = notch_filter(bp, freq=50.0, fs=fs)
    return min_max_normalize(notched)

def preprocess_ecg(raw_signal, fs=256):
    """
    ECG: Focuses on autonomic heart rate changes & QRS spikes (0.5 Hz - 40 Hz).
    """
    bp = butter_bandpass_filter(raw_signal, lowcut=0.5, highcut=min(40.0, 0.45 * fs), fs=fs)
    notched = notch_filter(bp, freq=50.0, fs=fs)
    return min_max_normalize(notched)

def preprocess_signal(raw_signal, fs=256, lowcut=10.0, highcut=100.0, notch_freq=50.0):
    """General preprocessing fallback."""
    bp_filtered = butter_bandpass_filter(raw_signal, lowcut=lowcut, highcut=highcut, fs=fs)
    notched = notch_filter(bp_filtered, freq=notch_freq, fs=fs)
    return min_max_normalize(notched)
