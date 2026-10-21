# Sleep Bruxism Detection using CWT and LSKNet

This repository contains the end-to-end deep learning framework for automated sleep bruxism detection from physiological signals (Chin EMG / EEG) using the CAP Sleep Database.

## Pipeline Overview
1. **Signal Preprocessing:**
   - 4th-order zero-phase Butterworth bandpass filter (10 Hz – 100 Hz / 250 Hz) to eliminate baseline wander and movement artifacts.
   - 50 Hz Notch filter to eliminate electrical power-line noise.
   - Min-Max amplitude normalization to scale signals to $[0, 1]$.
2. **Feature Extraction (CWT):**
   - Continuous Wavelet Transform (CWT) with Complex Morlet (`cmor1.5-1.0`) wavelets.
   - Conversion of 1D signal bursts into 2D time-frequency scalogram images (224x224 RGB).
3. **Deep Learning Model (LSKNet):**
   - Large Selective Kernel Network (LSKNet) backbone dynamically adjusts receptive fields to capture both brief clenches and sustained grinding episodes.
   - Baseline comparison with ResNet-18.

## Folder Structure
```text
sleep_bruxism_detection/
├── src/
│   ├── preprocessing.py    # Bandpass, Notch, and Normalization
│   ├── cwt_transform.py    # Continuous Wavelet Transform & Scalogram generator
│   ├── dataset.py          # EDF parsing with MNE & PyTorch Dataset
│   └── models.py           # LSKNet & ResNet-18 model builders
├── train.py                # GPU training & evaluation script
├── test_pipeline.py        # Local dry-run verification script
├── requirements.txt        # Python dependencies
└── README.md
```

## Quick Start (Local Verification)
To verify the signal processing and CWT generation on your laptop:
```bash
python test_pipeline.py
```

## Training on GPU System
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run training:
   ```bash
   python train.py --data_dir data/dataset --model_name lsknet_t --epochs 30 --batch_size 32
   ```
