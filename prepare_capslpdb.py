"""
Automated CAP Sleep Database Pipeline: Multi-Signal & Single-Signal Support.

KEY DESIGN:
- Step 1: Filter whole signal (bandpass + notch, NO normalization) → use for RMS burst detection.
- Step 2: Extract epoch windows.
- Step 3: Normalize each individual epoch window → then feed into CWT.
This two-step design preserves relative amplitude differences needed for detection,
while ensuring each CWT image has consistent [0,1] scale for model training.
"""

import os
import argparse
import urllib.request
import numpy as np
from tqdm import tqdm
import mne

from src.preprocessing import (
    filter_emg_only, filter_eeg_only, filter_ecg_only,
    preprocess_emg, preprocess_eeg, preprocess_ecg
)
from src.cwt_transform import generate_multisignal_scalogram_image

S3_BASE_URL = "https://physionet-open.s3.amazonaws.com/capslpdb/1.0.0/"
PHYSIONET_BASE_URL = "https://physionet.org/files/capslpdb/1.0.0/"

def download_file_fast(filename, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
        print(f"  [Exists] {dest_path} ({os.path.getsize(dest_path)/(1024*1024):.1f} MB). Skipping download.")
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    class Bar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    for url in [f"{S3_BASE_URL}{filename}", f"{PHYSIONET_BASE_URL}{filename}"]:
        try:
            print(f"  [Downloading] {url}")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with Bar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)
            return dest_path
        except Exception as e:
            print(f"  Warning: {url} failed ({e}). Trying fallback...")
    raise RuntimeError(f"Could not download {filename}.")

def find_channel(raw, candidates):
    for c in candidates:
        for ch in raw.ch_names:
            if c.lower() in ch.lower():
                return ch
    return None

def load_and_resample(edf_path, target_fs=256):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    emg_ch = find_channel(raw, ["EMG1-EMG2", "emg", "chin"])
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

def detect_and_extract(emg_raw, eeg_raw, ecg_raw, fs=256, epoch_sec=2.0, max_epochs=300):
    """
    TWO-STEP APPROACH:
    Step 1: Filter (no normalization) → compute RMS baseline on entire recording → detect bursts.
    Step 2: Extract epoch windows → normalize each window individually.
    """
    window_samples = int(epoch_sec * fs)

    # Step 1a: Filter whole signal — preserves amplitude ratios for detection
    print("  Step 1: Filtering whole signal for burst detection...")
    emg_filt = filter_emg_only(emg_raw, fs=fs)
    eeg_filt = filter_eeg_only(eeg_raw, fs=fs) if eeg_raw is not None else None
    ecg_filt = filter_ecg_only(ecg_raw, fs=fs) if ecg_raw is not None else None

    # Step 1b: Compute RMS across 2-second windows on filtered (unnormalized) EMG
    total_len = len(emg_filt)
    rms_vals = [
        np.sqrt(np.mean(emg_filt[i : i + window_samples] ** 2))
        for i in range(0, total_len - window_samples, window_samples)
    ]
    rms_arr = np.array(rms_vals)

    baseline_rms = np.percentile(rms_arr, 25)
    threshold = 2.0 * baseline_rms
    print(f"  Baseline RMS (25th pctile): {baseline_rms:.8f}")
    print(f"  Bruxism threshold (2x):     {threshold:.8f}")
    print(f"  Max window RMS:             {np.max(rms_arr):.8f}")
    print(f"  Windows above threshold:    {np.sum(rms_arr >= threshold)}")

    # Step 2: Extract windows + per-window normalization (ready for CWT)
    brux_epochs, norm_epochs = [], []
    for idx, i in enumerate(range(0, total_len - window_samples, window_samples)):
        if idx >= len(rms_arr):
            break
        rms = rms_arr[idx]

        # Raw window (filtered but not normalized yet)
        emg_win = emg_filt[i : i + window_samples]
        eeg_win = eeg_filt[i : i + window_samples] if eeg_filt is not None else None
        ecg_win = ecg_filt[i : i + window_samples] if ecg_filt is not None else None

        # Normalize each window individually for CWT
        emg_norm = preprocess_emg(emg_win, fs=fs)
        eeg_norm = preprocess_eeg(eeg_win, fs=fs) if eeg_win is not None else None
        ecg_norm = preprocess_ecg(ecg_win, fs=fs) if ecg_win is not None else None

        if rms >= threshold:
            brux_epochs.append((emg_norm, eeg_norm, ecg_norm))
        elif rms <= 1.3 * baseline_rms:
            norm_epochs.append((emg_norm, eeg_norm, ecg_norm))

    print(f"  Final: {len(brux_epochs)} bruxism bursts | {len(norm_epochs)} normal windows")
    return brux_epochs[:max_epochs], norm_epochs[:max_epochs]

def process_record(subject_id, raw_dir="data/raw_edf", output_dir="data/dataset", max_epochs=300, multisignal=True):
    edf_path = os.path.join(raw_dir, f"{subject_id}.edf")
    download_file_fast(f"{subject_id}.edf", edf_path)

    print(f"  Loading {edf_path}...")
    emg_raw, eeg_raw, ecg_raw, fs = load_and_resample(edf_path, target_fs=256)

    if emg_raw is None:
        print(f"  ERROR: No EMG channel found. Skipping {subject_id}.")
        return

    brux_epochs, norm_epochs = detect_and_extract(
        emg_raw,
        eeg_raw if multisignal else None,
        ecg_raw if multisignal else None,
        fs=fs,
        max_epochs=max_epochs
    )

    if len(brux_epochs) == 0:
        print(f"  WARNING: No bruxism bursts detected in {subject_id}!")
        return

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

    print("\n--- Dataset Summary ---")
    for split in ["train", "val"]:
        for cls in ["bruxism", "normal"]:
            d = os.path.join("data/dataset", split, cls)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            print(f"  data/dataset/{split}/{cls}/: {n} images")

    print("\nDone! Run: python3 train.py --model_name lsknet_tiny --epochs 30")

if __name__ == "__main__":
    main()
