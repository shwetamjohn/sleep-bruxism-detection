"""
Diagnostic script to inspect the EMG signal in brux1.edf and brux2.edf.
Run this BEFORE prepare_capslpdb.py to understand why bruxism bursts aren't being detected.
"""

import numpy as np
import mne
import os

def diagnose(edf_path, target_fs=256):
    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {edf_path}")
    print(f"{'='*60}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    print(f"\nChannels in this file:")
    for ch in raw.ch_names:
        print(f"  {ch}")
        
    fs_orig = raw.info['sfreq']
    print(f"\nOriginal sampling rate: {fs_orig} Hz")
    
    # Find EMG channel
    emg_ch = None
    for ch in raw.ch_names:
        if "emg" in ch.lower():
            emg_ch = ch
            break
    
    if emg_ch is None:
        print("ERROR: No EMG channel found!")
        return
        
    print(f"EMG channel identified: {emg_ch}")
    
    data = raw[emg_ch][0][0]
    print(f"\nRaw EMG Signal Stats:")
    print(f"  Length     : {len(data)} samples = {len(data)/fs_orig/3600:.2f} hours")
    print(f"  Mean       : {np.mean(data):.6f}")
    print(f"  Std Dev    : {np.std(data):.6f}")
    print(f"  Min        : {np.min(data):.6f}")
    print(f"  Max        : {np.max(data):.6f}")
    print(f"  Abs Mean   : {np.mean(np.abs(data)):.6f}")
    
    # Check RMS across 2-second windows
    win = int(2 * fs_orig)
    rms_vals = [np.sqrt(np.mean(data[i:i+win]**2)) for i in range(0, len(data)-win, win)]
    rms_arr = np.array(rms_vals)
    
    baseline_rms = np.percentile(rms_arr, 25)
    threshold_2x = 2.0 * baseline_rms
    threshold_1_5x = 1.5 * baseline_rms
    threshold_1_2x = 1.2 * baseline_rms
    
    print(f"\nRMS Analysis across {len(rms_vals)} windows (2s each):")
    print(f"  25th percentile (baseline)  : {baseline_rms:.8f}")
    print(f"  Median RMS                  : {np.median(rms_arr):.8f}")
    print(f"  75th percentile             : {np.percentile(rms_arr, 75):.8f}")
    print(f"  95th percentile             : {np.percentile(rms_arr, 95):.8f}")
    print(f"  Max RMS                     : {np.max(rms_arr):.8f}")
    
    print(f"\nDetection at different thresholds:")
    print(f"  >= 2.0x baseline ({threshold_2x:.8f}): {np.sum(rms_arr >= threshold_2x)} windows")
    print(f"  >= 1.5x baseline ({threshold_1_5x:.8f}): {np.sum(rms_arr >= threshold_1_5x)} windows")
    print(f"  >= 1.2x baseline ({threshold_1_2x:.8f}): {np.sum(rms_arr >= threshold_1_2x)} windows")

if __name__ == "__main__":
    for name in ["brux1.edf", "brux2.edf"]:
        path = os.path.join("data/raw_edf", name)
        if os.path.exists(path):
            diagnose(path)
        else:
            print(f"\nFile not found: {path}. Skipping.")
    
    print("\n[Done] Use the detection counts above to choose the right threshold in prepare_capslpdb.py.")
