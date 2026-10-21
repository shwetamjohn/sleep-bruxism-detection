"""
Automated CAP Sleep Database Pipeline: Download, Extract, Preprocess, and Generate CWT Dataset.

This script:
1. Downloads patient records from PhysioNet CAP Sleep Database (e.g. brux1, brux2, n1, n2).
2. Reads the Chin EMG biopotential channels using MNE.
3. Automatically detects Bruxism bursts using standard AASM (American Academy of Sleep Medicine)
   clinical criteria: Chin EMG elevations >= 2x baseline tonic activity lasting 0.5s - 3.0s.
4. Preprocesses each segment (10-100Hz Bandpass + 50Hz Notch + Min-Max normalization).
5. Computes Continuous Wavelet Transform (CWT) scalograms and saves them into:
   data/dataset/train/bruxism/
   data/dataset/train/normal/
   data/dataset/val/bruxism/
   data/dataset/val/normal/
"""

import os
import sys
import urllib.request
import numpy as np
from tqdm import tqdm
import mne

from src.preprocessing import preprocess_signal
from src.cwt_transform import generate_cwt_scalogram, scalogram_to_image

BASE_URL = "https://physionet.org/files/capslpdb/1.0.0/"

def download_file_with_progress(url, dest_path):
    """Downloads a file from PhysioNet with a progress bar."""
    if os.path.exists(dest_path):
        print(f"[Exists] {dest_path} already exists. Skipping download.")
        return dest_path
        
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[Downloading] {url} -> {dest_path}")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(dest_path)) as t:
        urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)
    return dest_path

def find_chin_emg_channel(raw):
    """Finds the chin / submental EMG channel name."""
    candidates = ["emg", "chin", "submentalis", "myo", "emg1", "emg2"]
    for ch in raw.ch_names:
        for c in candidates:
            if c in ch.lower():
                return ch
    # Fallback
    return raw.ch_names[0]

def extract_aasm_bruxism_segments(emg_signal, fs, epoch_sec=2.0):
    """
    Detects Bruxism bursts vs Normal baseline segments using clinical AASM criteria:
    - Calculates the running RMS baseline amplitude of the Chin EMG.
    - Class 1 (Bruxism): Signal amplitude elevates to >= 2x baseline tonic EMG.
    - Class 0 (Normal): Baseline calm muscle activity during stable sleep.
    """
    window_samples = int(epoch_sec * fs)
    total_len = len(emg_signal)
    
    # Calculate baseline tonic EMG level (10th to 25th percentile of RMS amplitude)
    step_samples = window_samples // 2
    rms_values = []
    for i in range(0, total_len - window_samples, step_samples):
        chunk = emg_signal[i : i + window_samples]
        rms_values.append(np.sqrt(np.mean(chunk**2)))
        
    rms_values = np.array(rms_values)
    baseline_rms = np.percentile(rms_values, 25)
    bruxism_threshold = 2.0 * baseline_rms
    
    bruxism_epochs = []
    normal_epochs = []
    
    for i in range(0, total_len - window_samples, window_samples):
        chunk = emg_signal[i : i + window_samples]
        chunk_rms = np.sqrt(np.mean(chunk**2))
        
        # Clinical condition: Burst > 2x baseline is Bruxism
        if chunk_rms >= bruxism_threshold:
            bruxism_epochs.append(chunk)
        elif chunk_rms <= 1.2 * baseline_rms:
            normal_epochs.append(chunk)
            
    return bruxism_epochs, normal_epochs

def process_record(subject_id, raw_dir="data/raw_edf", output_dir="data/dataset", max_epochs_per_class=300):
    """Downloads an EDF, extracts EMG, cleans it, and generates CWT scalograms."""
    edf_filename = f"{subject_id}.edf"
    url = f"{BASE_URL}{edf_filename}"
    edf_path = os.path.join(raw_dir, edf_filename)
    
    # 1. Download EDF
    download_file_with_progress(url, edf_path)
    
    # 2. Read EDF with MNE
    print(f"Reading {edf_path}...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    ch_name = find_chin_emg_channel(raw)
    fs = int(raw.info['sfreq'])
    print(f"Loaded channel: {ch_name} (Sampling Rate: {fs} Hz)")
    
    signal_data = raw[ch_name][0][0]
    
    # 3. Clean signal with Bandpass and Notch
    print("Filtering signal...")
    clean_signal = preprocess_signal(signal_data, fs=fs, lowcut=10.0, highcut=100.0, notch_freq=50.0)
    
    # 4. Extract Bruxism bursts and Normal epochs
    brux_epochs, norm_epochs = extract_aasm_bruxism_segments(clean_signal, fs=fs, epoch_sec=2.0)
    print(f"Found {len(brux_epochs)} bruxism burst candidates and {len(norm_epochs)} normal baseline candidates.")
    
    # Cap to max_epochs_per_class to balance classes and keep training fast
    brux_epochs = brux_epochs[:max_epochs_per_class]
    norm_epochs = norm_epochs[:max_epochs_per_class]
    
    # 5. Generate and save CWT Scalograms (80% train, 20% val)
    for class_name, epochs in [("bruxism", brux_epochs), ("normal", norm_epochs)]:
        n_train = int(len(epochs) * 0.8)
        
        for idx, ep in enumerate(tqdm(epochs, desc=f"Generating CWT for {class_name}")):
            split = "train" if idx < n_train else "val"
            dest_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            img_path = os.path.join(dest_dir, f"{subject_id}_{class_name}_{idx:04d}.png")
            if not os.path.exists(img_path):
                power, _ = generate_cwt_scalogram(ep, fs=fs, total_scales=64)
                img = scalogram_to_image(power, img_size=(224, 224), colormap="jet")
                img.save(img_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download CAP Sleep Database and generate CWT Scalograms")
    parser.add_argument("--subjects", nargs="+", default=["brux1", "n1"], help="List of subjects to download and process (e.g. brux1 brux2 n1 n2)")
    parser.add_argument("--max_epochs", type=int, default=250, help="Max scalogram images per class per subject")
    args = parser.parse_args()
    
    for subj in args.subjects:
        print(f"\n==========================================")
        print(f"  Processing Subject: {subj}")
        print(f"==========================================")
        process_record(subj, max_epochs_per_class=args.max_epochs)
        
    print("\nDataset preparation complete! All CWT scalogram images are saved in data/dataset/train/ and data/dataset/val/.")
    print("You can now run: python train.py --model_name lsknet_t --epochs 30")

if __name__ == "__main__":
    main()
