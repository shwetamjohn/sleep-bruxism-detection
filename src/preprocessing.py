"""
Signal Preprocessing Module for Sleep Bruxism Detection
Provides separate functions for:
  - filter_only: Bandpass + Notch (NO normalization) — used for burst detection
  - preprocess_*: Bandpass + Notch + Min-Max — used per-epoch before CWT
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def butter_bandpass_filter(data, lowcut=10.0, highcut=100.0, fs=256, order=4):
    nyquist = 0.5 * fs
    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, freq=50.0, fs=256, Q=30):
    if freq >= (0.5 * fs):
        return data
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data)

def min_max_normalize(data):
    d_min = np.min(data)
    d_max = np.max(data)
    if d_max - d_min == 0:
        return np.zeros_like(data)
    return (data - d_min) / (d_max - d_min + 1e-8)

# -----------------------------------------------------------------------
# FILTER ONLY (no normalization) — used for burst detection on whole signal
# -----------------------------------------------------------------------

def filter_emg_only(raw_signal, fs=256):
    """
    Bandpass (10-100 Hz) + 50 Hz Notch only. No normalization.
    Use this for computing RMS-based burst detection across the full recording.
    """
    bp = butter_bandpass_filter(raw_signal, lowcut=10.0, highcut=min(100.0, 0.45 * fs), fs=fs)
    return notch_filter(bp, freq=50.0, fs=fs)

def filter_eeg_only(raw_signal, fs=256):
    """Bandpass (0.5-35 Hz) + 50 Hz Notch. No normalization."""
    bp = butter_bandpass_filter(raw_signal, lowcut=0.5, highcut=min(35.0, 0.45 * fs), fs=fs)
    return notch_filter(bp, freq=50.0, fs=fs)

def filter_ecg_only(raw_signal, fs=256):
    """Bandpass (0.5-40 Hz) + 50 Hz Notch. No normalization."""
    bp = butter_bandpass_filter(raw_signal, lowcut=0.5, highcut=min(40.0, 0.45 * fs), fs=fs)
    return notch_filter(bp, freq=50.0, fs=fs)

# -----------------------------------------------------------------------
# PREPROCESS (filter + normalize) — used per-epoch window before CWT
# -----------------------------------------------------------------------

def preprocess_emg(epoch, fs=256):
    """Filter + Min-Max normalize a single epoch for CWT input."""
    filtered = filter_emg_only(epoch, fs=fs)
    return min_max_normalize(filtered)

def preprocess_eeg(epoch, fs=256):
    filtered = filter_eeg_only(epoch, fs=fs)
    return min_max_normalize(filtered)

def preprocess_ecg(epoch, fs=256):
    filtered = filter_ecg_only(epoch, fs=fs)
    return min_max_normalize(filtered)

def preprocess_signal(raw_signal, fs=256, lowcut=10.0, highcut=100.0, notch_freq=50.0):
    """Generic preprocessing fallback."""
    bp = butter_bandpass_filter(raw_signal, lowcut=lowcut, highcut=highcut, fs=fs)
    notched = notch_filter(bp, freq=notch_freq, fs=fs)
    return min_max_normalize(notched)
