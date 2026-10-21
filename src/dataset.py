"""
Dataset and Data Loading Module for Sleep Bruxism Detection
Supports reading EDF files with MNE, windowing/segmentation, and PyTorch Dataset creation.
"""

import os
import glob
import numpy as np
from PIL import Image

try:
    import mne
except ImportError:
    mne = None

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    # Fallback placeholder if torch is not installed locally
    class Dataset:
        pass

class BruxismScalogramDataset(Dataset):
    """
    PyTorch Dataset to load 2D CWT scalogram images organized in folders:
        data_dir/
            bruxism/  (Class 1)
            normal/   (Class 0)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Class 1: Bruxism
        brux_dir = os.path.join(root_dir, "bruxism")
        if os.path.isdir(brux_dir):
            for f in glob.glob(os.path.join(brux_dir, "*.png")):
                self.samples.append((f, 1))
                
        # Class 0: Normal
        norm_dir = os.path.join(root_dir, "normal")
        if os.path.isdir(norm_dir):
            for f in glob.glob(os.path.join(norm_dir, "*.png")):
                self.samples.append((f, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def extract_channel_from_edf(edf_path, channel_keywords=("emg", "chin", "submentalis")):
    """
    Finds and extracts the EMG channel from an EDF file using MNE.
    """
    if mne is None:
        raise ImportError("MNE is required to read EDF files. Please install it using `pip install mne`.")
        
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    selected_ch = None
    
    for ch in raw.ch_names:
        for kw in channel_keywords:
            if kw.lower() in ch.lower():
                selected_ch = ch
                break
        if selected_ch:
            break
            
    if selected_ch is None:
        # Default to first channel if not found
        selected_ch = raw.ch_names[0]
        print(f"Warning: Specific EMG channel not found in {edf_path}. Falling back to '{selected_ch}'.")
        
    data, times = raw[selected_ch]
    fs = int(raw.info['sfreq'])
    return data[0], fs, selected_ch

def slice_into_epochs(signal, fs, epoch_duration_sec=3.0, step_sec=3.0):
    """
    Slices a continuous 1D signal into equal-length epoch segments.
    """
    epoch_len = int(fs * epoch_duration_sec)
    step_len = int(fs * step_sec)
    total_samples = len(signal)
    
    epochs = []
    for start in range(0, total_samples - epoch_len + 1, step_len):
        epoch = signal[start : start + epoch_len]
        epochs.append(epoch)
        
    return np.array(epochs)
