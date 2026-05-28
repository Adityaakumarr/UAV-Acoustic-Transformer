# Audio Array-Based 3D UAV Trajectory Estimation with Transformer Networks

A research-oriented deep learning system for estimating **3D UAV trajectories** using **multi-channel audio recordings** from microphone arrays.

This project implements a complete end-to-end acoustic localization pipeline using:

- ROS bag audio extraction
- GCC-PHAT spatial acoustic processing
- CNN + Transformer trajectory modeling
- Ground-truth synchronization
- Velocity-aware trajectory learning
- 3D localization evaluation and visualization

---

# Features

- Multi-channel ROS1 audio extraction
- MMAUD dataset integration
- STFT + GCC-PHAT feature generation
- Full 6-pair GCC spatial encoding
- CNN acoustic feature encoder
- Transformer temporal trajectory modeling
- Velocity consistency loss
- Persistent train/inference normalization
- Accurate GT timestamp synchronization
- Full training / evaluation pipeline
- 3D trajectory visualization
- Standalone inference support

---

# Project Pipeline

```text
ROS Bags
   ↓
Audio Extraction
   ↓
Feature Extraction
   ↓
Ground Truth Alignment
   ↓
Transformer Training
   ↓
Evaluation
   ↓
Visualization / Inference
```

---

# Dataset Structure

```text
dataset/
├── bags/
│   ├── Pham4.bag
│   └── 2023-08-24-11-30-56_phantom4.bag
│
└── Pham4/
    └── ground_truth/
        ├── 1692847902.611685.npy
        ├── 1692847902.663781.npy
        └── ...
```

Each `.npy` file contains:

```python
[x, y, z]
```

The filename itself acts as the trajectory timestamp.

---

# Project Structure

```text
.
├── config.yaml
├── dataset.py
├── evaluate.py
├── extract_audio.py
├── feature_extraction.py
├── generate_aligned_labels.py
├── inference.py
├── model.py
├── train.py
├── visualize.py
├── requirements.txt
├── README.md
│
├── audio/
├── checkpoints/
├── dataset/
├── features/
├── logs/
└── outputs/
```

---

# Installation

## 1. Clone Repository

```bash
git clone <your_repo_url>
cd <repo_name>
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / Mac

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Main Dependencies

- torch
- torchvision
- numpy
- scipy
- pandas
- librosa
- matplotlib
- soundfile
- tqdm
- rosbags
- scikit-learn

---

# Feature Representation

Each extracted acoustic feature tensor has shape:

```text
(10, 513, 64)
```

Where:

| Channels | Description |
|---|---|
| 4 | Log-magnitude spectrograms |
| 6 | GCC-PHAT microphone-pair features |

---

# GCC-PHAT Microphone Pairs

The system uses all 6 microphone-pair combinations:

```python
[
    (0,1),
    (0,2),
    (0,3),
    (1,2),
    (1,3),
    (2,3)
]
```

This enables complete spatial acoustic encoding.

---

# Usage

# 1. Extract Audio from ROS Bags

```bash
python extract_audio.py
```

Output:

```text
audio/
└── Pham4/
    ├── ch1.wav
    ├── ch2.wav
    ├── ch3.wav
    └── ch4.wav
```

---

# 2. Extract Acoustic Features

```bash
python feature_extraction.py
```

Output:

```text
features/
└── Pham4/
    ├── window_00000.npy
    ├── window_00001.npy
    └── ...
```

Feature tensor shape:

```text
(10, 513, 64)
```

---

# 3. Generate Time-Aligned Labels

```bash
python generate_aligned_labels.py
```

This synchronizes:
- audio windows
- GT timestamps

and generates:

```text
features/Pham4/labels.csv
```

---

# 4. Train Model

```bash
python train.py
```

Generated checkpoints:

```text
checkpoints/
├── best_model.pth
├── last_model.pth
└── normalizer.pt
```

---

# 5. Evaluate Model

```bash
python evaluate.py
```

Outputs:

```text
outputs/
├── metrics.csv
└── predictions.csv
```

---

# 6. Visualize Results

```bash
python visualize.py --eval_csv outputs/predictions.csv
```

Generated plots:

```text
outputs/
├── trajectory_3d.png
├── trajectory_axes.png
├── trajectory_side_by_side.png
└── error_histogram.png
```

---

# 7. Run Standalone Inference

```bash
python inference.py --audio_dir audio/Pham4
```

Outputs:

```text
outputs/
├── predicted_trajectory.csv
└── trajectory_plot.png
```

---

# Model Architecture

## Acoustic Encoder
- Multi-channel log spectrograms
- GCC-PHAT spatial features
- CNN feature extraction

## Temporal Modeling
- Transformer encoder
- Positional encoding
- Trajectory-conditioned decoding

## Training Improvements
- Persistent feature normalization
- Velocity consistency loss
- Gradient clipping
- Cosine annealing scheduler
- Early stopping

## Output
- 3D UAV position regression
- `(x, y, z)` coordinates

---

# Final Evaluation Metrics

| Metric | Value |
|---|---|
| APE | 0.196 m |
| Dx | 0.134 m |
| Dy | 0.119 m |
| Dz | 0.027 m |

---

# Baseline Comparison (APE)

| Method | APE (m) |
|---|---|
| AudioNet | 2.80 |
| DroneChase | 2.64 |
| TAME | 0.55 |
| Ours | 0.20 |

---

# Visualization Outputs

The system generates:

- 3D trajectory overlays
- X/Y/Z temporal plots
- Error histograms
- GT vs prediction comparisons

---

# Configuration

Main configuration file:

```text
config.yaml
```

Controls:
- audio preprocessing
- feature extraction
- model architecture
- training hyperparameters
- dataset paths
- evaluation settings

---

# Current Status

Implemented:
- End-to-end acoustic UAV trajectory estimation
- MMAUD dataset integration
- Full GCC spatial processing
- GT synchronization
- Persistent normalization
- Velocity-aware trajectory learning
- Stable Transformer training
- Evaluation / inference / visualization pipeline

Current Performance:
- ~0.20 m Absolute Position Error (APE)

---

# Future Enhancements

Planned future work:

- GT interpolation
- Temporal smoothing
- Real-time streaming inference
- Multi-sequence training
- Acoustic robustness experiments
- Noise-condition evaluation
- LiDAR pseudo-label integration

---

# Research Notes

This project demonstrates that acoustic localization can provide:

- illumination-invariant UAV tracking
- robust spatial trajectory estimation
- strong performance in low-visibility conditions

without relying on camera-based tracking.

---

# Citation

If you use this project in research, please cite the original MMAUD and UAV acoustic localization papers accordingly.

---

# License

MIT License

---

# Author

Shubham Sharma

Research Project — Audio-Based UAV 3D Trajectory Estimation