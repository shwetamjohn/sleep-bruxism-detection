"""
Local Laptop Verification Script (Multi-Signal & Single-Signal)
Tests the full signal processing and Multi-Signal CWT scalogram generation pipeline.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import preprocess_emg, preprocess_eeg, preprocess_ecg
from src.cwt_transform import generate_multisignal_scalogram_image

def run_local_test():
    print("==================================================")
    print("  Sleep Bruxism Detection - Multi-Signal Test    ")
    print("==================================================")
    
    fs = 256
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(fs * duration_sec), endpoint=False)
    
    # 1. Simulate 3 synchronized signals
    # EMG: Muscle burst (grind) from 0.8s to 1.6s
    emg = 0.1 * np.random.randn(len(t))
    emg[(t >= 0.8) & (t <= 1.6)] += 1.8 * np.sin(2 * np.pi * 40 * t[(t >= 0.8) & (t <= 1.6)])
    
    # EEG: Brain micro-arousal (alpha/theta burst just before & during clench)
    eeg = 0.2 * np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.randn(len(t))
    eeg[(t >= 0.5) & (t <= 1.2)] += 1.2 * np.sin(2 * np.pi * 10 * t[(t >= 0.5) & (t <= 1.2)])
    
    # ECG: Cardiac QRS wave spike
    ecg = 0.1 * np.random.randn(len(t))
    ecg[int(0.5 * fs) : int(0.55 * fs)] += 2.0
    ecg[int(1.5 * fs) : int(1.55 * fs)] += 2.0
    
    print(f"[Step 1] Simulated 3 synchronized signals: EMG (chin), EEG (brain), ECG (heart).")
    
    # 2. Preprocessing
    emg_c = preprocess_emg(emg, fs=fs)
    eeg_c = preprocess_eeg(eeg, fs=fs)
    ecg_c = preprocess_ecg(ecg, fs=fs)
    print(f"[Step 2] Filtered all 3 signals with specialized bandpass & notch filters.")
    
    # 3. Multi-Signal CWT Scalogram (3 Channels: R=EMG, G=EEG, B=ECG)
    os.makedirs("sample_output", exist_ok=True)
    multi_img = generate_multisignal_scalogram_image(emg_c, eeg_c, ecg_c, fs=fs, total_scales=64, img_size=(224, 224))
    multi_path = os.path.join("sample_output", "multisignal_scalogram.png")
    multi_img.save(multi_path)
    print(f"[Step 3] Generated 3-Channel Multi-Signal Scalogram Image!")
    print(f"         Saved at: {multi_path}")
    print(f"         Dimensions: {multi_img.size}, Mode: {multi_img.mode}")
    print(f"         Channel R = EMG (Jaw Clench)")
    print(f"         Channel G = EEG (Brain Micro-Arousal)")
    print(f"         Channel B = ECG (Cardiac Response)")
    
    print("\n--------------------------------------------------")
    print("MULTI-SIGNAL PIPELINE PASSED SUCCESSFULLY!")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    run_local_test()
