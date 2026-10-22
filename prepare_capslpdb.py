"""
Automated CAP Sleep Database Pipeline: Multi-Signal & Single-Signal Support.

Key fix: Bandpass filter is applied BEFORE computing the baseline RMS threshold.
Raw EDF EMG signals have a large DC offset/baseline wander that inflates the
25th percentile baseline, making 2x threshold impossible to breach.
"""

import os
import sys
import argparse
import urllib.request
import numpy as np
from tqdm import tqdm
import mne

from src.preprocessing import preprocess_emg, preprocess_eeg, preprocess_ecg
from src.cwt_transform import generate_multisignal_scalogram_image

S3_BASE_URL = "https://physionet-open.s3.amazonaws.com/capslpdb/1.0.0/"
PHYSIONET_BASE_URL = "https://physionet.org/files/capslpdb/1.0.0/"

def download_file_fast(filename, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
        print(f"[Exists] {dest_path} ({os.path.getsize(dest_path)/(1024*1024):.1f} MB). Skipping download.")
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    class Bar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
            
    for url in [f"{S3_BASE_URL}{filename}", f"{PHYSIONET_BASE_URL}{filename}"]:
        try:
            print(f"[Downloading] {url}")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with Bar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)
            return dest_path
        except Exception as e:
            print(f"Warning: {url} failed ({e}). Trying next mirror...")
    raise RuntimeError(f"Could not download {filename}.")

def find_channel(raw, candidates):
    for c in candidates:
        for ch in raw.ch_names:
            if c.lower() in ch.lower():
                return ch
    return None

def extract_and_resample(raw, target_fs=256):
    emg_ch = find_channel(raw, ["EMG1-EMG2", "emg", "chin", "submentalis"])
    eeg_ch = find_channel(raw, ["C4-A1", "C4-P4", "F4-C4", "FP1-F3"])
    ecg_ch = find_channel(raw, ["ECG1-ECG2", "ecg"])
    
    print(f"  Channels: EMG='{emg_ch}', EEG='{eeg_ch}', ECG='{ecg_ch}'")
    
    if raw.info['sfreq'] != target_fs:
        print(f"  Resampling {raw.info['sfreq']} Hz -> {target_fs} Hz...")
        raw = raw.copy().resample(target_fs)
        
    emg = raw[emg_ch][0][0] if emg_ch else None
    eeg = raw[eeg_ch][0][0] if eeg_ch else None
    ecg = raw[ecg_ch][0][0] if ecg_ch else None
    
    return emg, eeg, ecg, int(raw.info['sfreq'])

def detect_and_extract(emg_clean, eeg_clean, ecg_clean, fs=256, epoch_sec=2.0, max_epochs=300):
    """
    Detects bruxism bursts on the ALREADY FILTERED (clean) EMG signal.
    Because we filter before computing RMS, the baseline is now accurate.
    Uses 2x baseline as bruxism threshold (clinical AASM standard).
    """
    window_samples = int(epoch_sec * fs)
    total_len = len(emg_clean)
    
    # Compute RMS for each 2-second window on the FILTERED signal
    rms_vals = []
    for i in range(0, total_len - window_samples, window_samples):
        chunk = emg_clean[i : i + window_samples]
        rms_vals.append(np.sqrt(np.mean(chunk ** 2)))
    rms_arr = np.array(rms_vals)
    
    # Baseline = 25th percentile of RMS (calm background muscle tone)
    baseline_rms = np.percentile(rms_arr, 25)
    threshold = 2.0 * baseline_rms
    
    print(f"  EMG baseline RMS (25th pctile): {baseline_rms:.6f}")
    print(f"  Bruxism threshold (2x):         {threshold:.6f}")
    print(f"  Max window RMS:                 {np.max(rms_arr):.6f}")
    
    brux, normal = [], []
    for idx, i in enumerate(range(0, total_len - window_samples, window_samples)):
        emg_c = emg_clean[i : i + window_samples]
        eeg_c = eeg_clean[i : i + window_samples] if eeg_clean is not None else None
        ecg_c = ecg_clean[i : i + window_samples] if ecg_clean is not None else None
        
        rms = rms_arr[idx] if idx < len(rms_arr) else 0
        if rms >= threshold:
            brux.append((emg_c, eeg_c, ecg_c))
        elif rms <= 1.3 * baseline_rms:
            normal.append((emg_c, eeg_c, ecg_c))
            
    print(f"  Detected: {len(brux)} bruxism bursts, {len(normal)} normal windows")
    return brux[:max_epochs], normal[:max_epochs]

def process_record(subject_id, raw_dir="data/raw_edf", output_dir="data/dataset", max_epochs=300, multisignal=True):
    edf_path = os.path.join(raw_dir, f"{subject_id}.edf")
    download_file_fast(f"{subject_id}.edf", edf_path)
    
    print(f"\n  Loading {edf_path}...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    emg, eeg, ecg, fs = extract_and_resample(raw, target_fs=256)
    
    # CRITICAL FIX: Apply bandpass filter BEFORE baseline/threshold computation
    print("  Filtering signals (bandpass + notch)...")
    emg_c = preprocess_emg(emg, fs=fs) if emg is not None else None
    eeg_c = preprocess_eeg(eeg, fs=fs) if (eeg is not None and multisignal) else None
    ecg_c = preprocess_ecg(ecg, fs=fs) if (ecg is not None and multisignal) else None
    
    if emg_c is None:
        print(f"  ERROR: No EMG channel found in {subject_id}.edf. Skipping.")
        return
    
    # Detect on filtered signal
    brux_epochs, norm_epochs = detect_and_extract(emg_c, eeg_c, ecg_c, fs=fs, max_epochs=max_epochs)
    
    if len(brux_epochs) == 0:
        print(f"  WARNING: No bruxism bursts detected in {subject_id}. Check diagnose_emg.py output.")
        return
    
    # Generate and save CWT images
    for class_name, epochs in [("bruxism", brux_epochs), ("normal", norm_epochs)]:
        n_train = int(len(epochs) * 0.8)
        for idx, (e_emg, e_eeg, e_ecg) in enumerate(tqdm(epochs, desc=f"  CWT [{subject_id}] {class_name}")):
            split = "train" if idx < n_train else "val"
            dest = os.path.join(output_dir, split, class_name)
            os.makedirs(dest, exist_ok=True)
            img_path = os.path.join(dest, f"{subject_id}_{class_name}_{idx:04d}.png")
            if not os.path.exists(img_path):
                img = generate_multisignal_scalogram_image(
                    emg_epoch=e_emg,
                    eeg_epoch=e_eeg if multisignal else None,
                    ecg_epoch=e_ecg if multisignal else None,
                    fs=fs, total_scales=64, img_size=(224, 224)
                )
                img.save(img_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=["brux1", "brux2"])
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--single_signal", action="store_true")
    args = parser.parse_args()
    
    multisignal = not args.single_signal
    mode = "MULTI-SIGNAL (EMG+EEG+ECG)" if multisignal else "SINGLE-SIGNAL (EMG)"
    print(f"\n=== CAP Sleep Dataset Generator [{mode}] ===\n")
    
    for subj in args.subjects:
        print(f"\n--- Subject: {subj} ---")
        process_record(subj, max_epochs=args.max_epochs, multisignal=multisignal)
        
    # Summary
    for split in ["train", "val"]:
        for cls in ["bruxism", "normal"]:
            d = os.path.join("data/dataset", split, cls)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            print(f"  data/dataset/{split}/{cls}/: {n} images")
            
    print("\nDone! Run: python3 train.py --model_name lsknet_tiny --epochs 30")

if __name__ == "__main__":
    main()
