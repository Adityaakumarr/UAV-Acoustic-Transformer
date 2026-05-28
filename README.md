# 🚁 UAV 3D Trajectory Estimation via Acoustic Transformer

This repository contains the official implementation for reproducing the pipeline of the paper **"Listening To UAV 3D Trajectory Estimation Via Acoustic Transformer" (ICASSP 2026)**.

The project takes raw 4-channel acoustic data (from drone microphones) and uses a custom Transformer-based architecture to accurately predict the drone's 3D trajectory (x, y, z) in real-time.

## ✨ Features
- **Multi-Drone Support:** Dynamically processes and trains on datasets from multiple drones (Phantom 4, Mavic 2, Mavic 3) in a unified pipeline.
- **Acoustic Preprocessing:** Extracts dense 10-channel spatial acoustic features combining log-magnitude spectrograms (4 channels) and GCC-PHAT spatial correlations (6 microphone pairs).
- **Transformer Architecture:** Replaces traditional CNN-RNN architectures with a powerful multi-head Self-Attention Transformer encoder/decoder for superior temporal trajectory tracking.
- **Automated Pipeline:** Contains end-to-end scripts for audio extraction, timestamp alignment, training, and 3D visualization.

---

## 🛠️ Pipeline Setup & Usage

### 1. Data Preparation
Raw drone datasets are provided as `.bag` (ROS) and `.zip` files containing ground truth `.npy` coordinates.
1. Place `.bag` files in `dataset/bags/`.
2. Extract the ground truth zips into `dataset/<drone_name>/ground_truth/`.

### 2. Audio Extraction
Extract 4-channel `.wav` audio from the ROS `.bag` files:
```bash
python extract_audio.py --bag_dir dataset/bags --out_dir audio
```

### 3. Feature Processing (Spectrograms + GCC-PHAT)
Convert the extracted raw audio into 10-channel `.npy` acoustic feature tensors:
```bash
python feature_extraction.py
```
*Note: Uses the "Purge Protocol" to manage disk space by automatically clearing raw audio once processed.*

### 4. Label Synchronization
Time-align the 3D ground truth coordinates with the center of the acoustic feature windows:
```bash
# Example for Mavic 3
python generate_aligned_labels.py --sequence Mavic3
```

### 5. Training
Train the Acoustic Transformer model:
```bash
python train.py
```
*(Check `config.yaml` to adjust hyper-parameters like learning rate, batch size, and network dimensions).*

### 6. Evaluation & Visualization
Generate metrics (APE, Dx, Dy, Dz) and interactive publication-ready 3D trajectory plots:
```bash
python evaluate.py
python visualize_3d_pham4_paper.py
```

---

## 🚀 Google Colab Integration
This project is highly optimized for Google Colab GPU training. 
Instead of uploading millions of tiny `.npy` files to Google Drive, upload only the raw `.bag` and `.zip` datasets. A Colab script is provided to extract and build the dataset directly on Colab's high-speed local storage before initiating GPU training.

## 📄 License
This project is for academic and research purposes. Please refer to the original paper for citation details.
