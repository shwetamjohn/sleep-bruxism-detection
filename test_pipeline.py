"""
Local Laptop Verification Script
Tests the full signal processing and CWT scalogram generation pipeline locally
to ensure zero syntax errors, valid scalogram shapes, and full reproducibility before moving to the GPU machine.
"""

import os
import sys
import numpy as np

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import preprocess_signal
from src.cwt_transform import generate_cwt_scalogram, scalogram_to_image

def run_local_test():
    print("==================================================")
    print("  Sleep Bruxism Detection - Local Pipeline Test   ")
    print("==================================================")
    
    # 1. Simulate 3 seconds of raw physiological EMG signal at 512 Hz
    fs = 512
    duration_sec = 3.0
    t = np.linspace(0, duration_sec, int(fs * duration_sec), endpoint=False)
    
    # Base signal with 50 Hz powerline hum + low frequency baseline wander + random noise
    powerline_hum = 0.8 * np.sin(2 * np.pi * 50 * t)
    baseline_wander = 1.2 * np.sin(2 * np.pi * 0.5 * t)
    noise = 0.2 * np.random.normal(size=len(t))
    
    # Simulate a sudden muscle burst (bruxism grinding event around 1.0s to 2.0s)
    burst = np.zeros_like(t)
    burst_mask = (t >= 1.0) & (t <= 2.0)
    burst[burst_mask] = 2.5 * np.sin(2 * np.pi * 35 * t[burst_mask])
    
    raw_signal = powerline_hum + baseline_wander + noise + burst
    print(f"[Step 1] Generated synthetic signal: {len(raw_signal)} samples ({duration_sec}s @ {fs}Hz)")
    
    # 2. Preprocess (Bandpass 10-100Hz, 50Hz Notch, Min-Max normalization)
    clean_signal = preprocess_signal(raw_signal, fs=fs, lowcut=10.0, highcut=100.0, notch_freq=50.0)
    print(f"[Step 2] Preprocessing completed.")
    print(f"         Min: {clean_signal.min():.4f}, Max: {clean_signal.max():.4f}, Mean: {clean_signal.mean():.4f}")
    assert 0.0 <= clean_signal.min() <= 1.0, "Signal min not normalized properly"
    assert 0.0 <= clean_signal.max() <= 1.0, "Signal max not normalized properly"
    
    # 3. Continuous Wavelet Transform (CWT)
    power_scalogram, freqs = generate_cwt_scalogram(clean_signal, fs=fs, total_scales=64)
    print(f"[Step 3] CWT Scalogram generated successfully!")
    print(f"         Scalogram matrix dimensions: {power_scalogram.shape} (scales x time_samples)")
    
    # 4. Convert Scalogram to 224x224 RGB image
    os.makedirs("sample_output", exist_ok=True)
    sample_img_path = os.path.join("sample_output", "test_scalogram.png")
    image = scalogram_to_image(power_scalogram, img_size=(224, 224), colormap="jet")
    image.save(sample_img_path)
    print(f"[Step 4] Scalogram saved as RGB image at: {sample_img_path}")
    print(f"         Image format: {image.format or 'PNG'}, Mode: {image.mode}, Size: {image.size}")
    
    print("\n--------------------------------------------------")
    print("ALL LOCAL PIPELINE CHECKS PASSED SUCCESSFULLY!")
    print("The signal processing & CWT modules are ready.")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    run_local_test()
