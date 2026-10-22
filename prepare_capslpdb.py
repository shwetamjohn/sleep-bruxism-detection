"""
Automated CAP Sleep Database Pipeline: Multi-Signal & Single-Signal Support.

This script:
1. Downloads patient records from the high-speed AWS S3 mirror (brux1, brux2, n1, n2).
2. Reads and aligns multiple physiological channels using MNE:
   - Chin EMG: EMG1-EMG2 (muscle clench)
   - EEG: C4-A1 or C4-P4 (brain micro-arousal)
   - ECG: ECG1-ECG2 (cardiac autonomic response)
3. Resamples to standard 256 Hz.
4. Detects Bruxism bursts vs Normal baseline using clinical AASM criteria (EMG >= 2x baseline).
5. Preprocesses each modality with specialized bandpass and notch filters.
6. Computes Continuous Wavelet Transform (CWT) scalograms and stacks them into:
   - Multi-Signal Mode: 3-channel image (Red=EMG, Green=EEG, Blue=ECG)
   - Single-Signal Mode: 3-channel color-mapped EMG scalogram
7. Saves images ready for LSKNet training.
"""

import os
import sys
import argparse
import urllib.request
import numpy as np
from tqdm import tqdm
import mne

from src.preprocessing import preprocess_emg, preprocess_eeg, preprocess_ecg, preprocess_signal
from src.cwt_transform import generate_multisignal_scalogram_image

# High-speed AWS S3 mirror identified by teammate
S3_BASE_URL = "https://physionet-open.s3.amazonaws.com/capslpdb/1.0.0/"
PHYSIONET_BASE_URL = "https://physionet.org/files/capslpdb/1.0.0/"

def download_file_fast(filename, dest_path):
    """Downloads an EDF file using the fast AWS S3 mirror with progress bar."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000000:
        print(f"[Exists] {dest_path} already present ({os.path.getsize(dest_path)/(1024*1024):.1f} MB). Skipping.")
        return dest_path
        
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urls = [f"{S3_BASE_URL}{filename}", f"{PHYSIONET_BASE_URL}{filename}"]
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    for url in urls:
        try:
            print(f"[Downloading] {url} -> {dest_path}")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)
            return dest_path
        except Exception as e:
            print(f"Warning: Download from {url} failed: {e}. Trying fallback...")
            
    raise RuntimeError(f"Could not download {filename} from any mirror.")

def find_channel(raw, candidates):
    """Finds matching channel in raw EDF."""
    for c in candidates:
        for ch in raw.ch_names:
            if c.lower() in ch.lower():
                return ch
    return None

def extract_multisignal_channels(raw, target_fs=256):
    """
    Extracts Chin EMG, EEG, and ECG channels and resamples them to target_fs.
    """
    # 1. EMG (Chin)
    emg_ch = find_channel(raw, ["EMG1-EMG2", "emg", "chin", "submentalis"])
    # 2. EEG (Central/Frontal)
    eeg_ch = find_channel(raw, ["C4-A1", "C4-P4", "F4-C4", "FP1-F3", "eeg"])
    # 3. ECG (Cardiac)
    ecg_ch = find_channel(raw, ["ECG1-ECG2", "ecg", "heart"])
    
    print(f"  Selected Channels: EMG='{emg_ch}', EEG='{eeg_ch}', ECG='{ecg_ch}'")
    
    # Resample if needed
    if raw.info['sfreq'] != target_fs:
        print(f"  Resampling recording from {raw.info['sfreq']} Hz to {target_fs} Hz...")
        raw = raw.copy().resample(target_fs)
        
    emg_data = raw[emg_ch][0][0] if emg_ch else None
    eeg_data = raw[eeg_ch][0][0] if eeg_ch else None
    ecg_data = raw[ecg_ch][0][0] if ecg_ch else None
    
    return emg_data, eeg_data, ecg_data, target_fs

def extract_aligned_epochs(emg, eeg, ecg, fs=256, epoch_sec=2.0, max_epochs_per_class=300):
    """
    Detects Bruxism bursts on EMG (burst >= 2x baseline) and extracts synchronized
    windows for EMG, EEG, and ECG.
    """
    window_samples = int(epoch_sec * fs)
    total_len = len(emg)
    
    # Calculate baseline tonic EMG level (25th percentile of RMS)
    step = window_samples // 2
    rms_vals = [np.sqrt(np.mean(emg[i : i + window_samples]**2)) for i in range(0, total_len - window_samples, step)]
    baseline_rms = np.percentile(rms_vals, 25)
    bruxism_threshold = 2.0 * baseline_rms
    
    bruxism_samples = []
    normal_samples = []
    
    for i in range(0, total_len - window_samples, window_samples):
        emg_chunk = emg[i : i + window_samples]
        chunk_rms = np.sqrt(np.mean(emg_chunk**2))
        
        eeg_chunk = eeg[i : i + window_samples] if eeg is not None else None
        ecg_chunk = ecg[i : i + window_samples] if ecg is not None else None
        
        if chunk_rms >= bruxism_threshold:
            bruxism_samples.append((emg_chunk, eeg_chunk, ecg_chunk))
        elif chunk_rms <= 1.2 * baseline_rms:
            normal_samples.append((emg_chunk, eeg_chunk, ecg_chunk))
            
    print(f"  Detected candidates: {len(bruxism_samples)} Bruxism bursts, {len(normal_samples)} Normal sleep windows.")
    return bruxism_samples[:max_epochs_per_class], normal_samples[:max_epochs_per_class]

def process_and_save(subject_id, raw_dir="data/raw_edf", output_dir="data/dataset", max_epochs=300, multisignal=True):
    edf_filename = f"{subject_id}.edf"
    edf_path = os.path.join(raw_dir, edf_filename)
    
    # 1. Download
    download_file_fast(edf_filename, edf_path)
    
    # 2. Read EDF
    print(f"Loading {edf_path}...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # 3. Extract channels
    emg, eeg, ecg, fs = extract_multisignal_channels(raw, target_fs=256)
    
    # 4. Clean signals
    print("  Applying bandpass & notch filters to signals...")
    emg_clean = preprocess_emg(emg, fs=fs)
    eeg_clean = preprocess_eeg(eeg, fs=fs) if (eeg is not None and multisignal) else None
    ecg_clean = preprocess_ecg(ecg, fs=fs) if (ecg is not None and multisignal) else None
    
    # 5. Extract aligned epochs
    brux_epochs, norm_epochs = extract_aligned_epochs(emg_clean, eeg_clean, ecg_clean, fs=fs, max_epochs_per_class=max_epochs)
    
    # 6. Generate Scalogram Images
    mode_str = "Multi-Signal (EMG+EEG+ECG)" if multisignal else "Single-Signal (EMG)"
    print(f"  Generating {mode_str} CWT Scalogram images...")
    
    for class_name, epochs in [("bruxism", brux_epochs), ("normal", norm_epochs)]:
        n_train = int(len(epochs) * 0.8)
        for idx, (emg_c, eeg_c, ecg_c) in enumerate(tqdm(epochs, desc=f"  CWT for {class_name}")):
            split = "train" if idx < n_train else "val"
            dest_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            img_path = os.path.join(dest_dir, f"{subject_id}_{class_name}_{idx:04d}.png")
            if not os.path.exists(img_path):
                img = generate_multisignal_scalogram_image(
                    emg_epoch=emg_c,
                    eeg_epoch=eeg_c if multisignal else None,
                    ecg_epoch=ecg_c if multisignal else None,
                    fs=fs,
                    total_scales=64,
                    img_size=(224, 224)
                )
                img.save(img_path)

def main():
    parser = argparse.ArgumentParser(description="Download CAP Sleep Database and generate CWT Scalograms")
    parser.add_argument("--subjects", nargs="+", default=["brux1", "brux2"], help="Subjects to process (e.g. brux1 brux2 n1 n2)")
    parser.add_argument("--max_epochs", type=int, default=300, help="Max epochs per class per subject")
    parser.add_argument("--single_signal", action="store_true", help="If set, uses EMG only instead of Multi-Signal (EMG+EEG+ECG)")
    args = parser.parse_args()
    
    multisignal = not args.single_signal
    mode_name = "MULTI-SIGNAL (EMG + EEG + ECG)" if multisignal else "SINGLE-SIGNAL (EMG)"
    
    print("\n=======================================================")
    print(f"  CAP Sleep Dataset Generator - Mode: {mode_name}")
    print("=======================================================\n")
    
    for subj in args.subjects:
        print(f"\n--- Processing Subject: {subj} ---")
        process_and_save(subj, max_epochs=args.max_epochs, multisignal=multisignal)
        
    print("\nDataset preparation complete! Scalogram images ready in data/dataset/train and data/dataset/val.")
    print("Now run: python train.py --model_name lsknet_t --epochs 30")

if __name__ == "__main__":
    main()
