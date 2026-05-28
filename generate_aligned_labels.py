"""
generate_aligned_labels.py
──────────────────────────
Synchronizes and time-aligns processed acoustic window features with high-precision
3D spatial ground-truth trajectories via timestamp interpolation.

Fulfills Step 3 of the UAV 3D Trajectory Estimation pipeline.
"""

import os
import yaml
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Config & Paths
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def main() -> None:
    print("\n========== UAV TRAJECTORY LABEL ALIGNMENT ==========\n")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="Pham4", help="Name of the drone sequence (e.g., Pham4, Mavic3, Mavic2)")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Paths updated for multi-drone support
    dataset_root = Path(cfg["paths"]["dataset_root"])
    
    # Check both structure types: dataset/Mavic3/ground_truth OR dataset/pham4_dataset/Pham4/gt
    gt_dir_1 = dataset_root / args.sequence / "ground_truth"
    gt_dir_2 = Path(cfg["paths"].get("gt_dir", "")) # Fallback to config if needed
    
    # Auto-resolve the ground truth directory
    if gt_dir_1.exists():
        gt_dir = gt_dir_1
    else:
        # If specific drone gt_dir doesn't exist, try falling back to config paths (e.g. for pham4)
        gt_dir = Path(cfg["paths"]["gt_dir"])
        
    features_dir = Path(cfg["paths"]["features_dir"]) / args.sequence
    out_csv = features_dir / "labels.csv"
    
    sample_rate = cfg["audio"]["sample_rate"]
    win_ms = cfg["audio"]["window_ms"]
    ovlp = cfg["audio"]["overlap"]
    
    # Calculate window specifications
    window_samples = int(win_ms * 1e-3 * sample_rate)
    hop_samples = int(window_samples * (1 - ovlp))

    # ── Step 1: Load Ground Truth Poses & Timestamps ──────────────────────────
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")
        
    gt_files = sorted(gt_dir.glob("*.npy"))
    if not gt_files:
        raise FileNotFoundError(f"No ground-truth .npy files found in {gt_dir}")
        
    print(f"[1/4] Loading {len(gt_files)} ground-truth trajectory files...")
    
    gt_times = []
    gt_positions = []
    
    for file in gt_files:
        # File stem represents raw Unix timestamp in seconds
        timestamp = float(file.stem)
        xyz = np.load(file)  # [x, y, z]
        gt_times.append(timestamp)
        gt_positions.append(xyz)
        
    gt_times = np.array(gt_times)
    gt_positions = np.array(gt_positions)
    
    # Sort by timestamp to ensure correct interpolation behavior
    sort_idx = np.argsort(gt_times)
    gt_times = gt_times[sort_idx]
    gt_positions = gt_positions[sort_idx]
    
    start_time = gt_times[0]
    end_time = gt_times[-1]
    duration = end_time - start_time
    
    print(f"  SUCCESS: Trajectory Start Time : {start_time:.6f}")
    print(f"  SUCCESS: Trajectory End Time   : {end_time:.6f}")
    print(f"  SUCCESS: Total Duration        : {duration:.2f} seconds")

    # ── Step 2: Map Processed Acoustic Feature Windows ────────────────────────
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
        
    feat_files = sorted(features_dir.glob("window_*.npy"))
    num_windows = len(feat_files)
    
    if num_windows == 0:
        raise FileNotFoundError(f"No processed window_*.npy files found in {features_dir}")
        
    print(f"\n[2/4] Found {num_windows} processed acoustic feature window(s) inside {features_dir}")

    # ── Step 3: Perform High-Precision Trajectory Interpolation ────────────────
    print(f"\n[3/4] Interpolating 3D trajectory positions for each window center...")
    
    rows = []
    
    for idx in range(num_windows):
        # Precise window center offset in samples relative to audio start
        center_sample = idx * hop_samples + window_samples / 2
        
        # Window absolute center time (seconds since Unix Epoch)
        # Synchronization anchor: Audio starts at the first pose timestamp (start_time)
        center_time = start_time + center_sample / sample_rate
        
        # High-precision linear interpolation across the trajectory timeline.
        # np.interp clamps values to boundary values, preventing out-of-bounds index leakage.
        x = np.interp(center_time, gt_times, gt_positions[:, 0])
        y = np.interp(center_time, gt_times, gt_positions[:, 1])
        z = np.interp(center_time, gt_times, gt_positions[:, 2])
        
        # Corresponding feature file path
        feat_filename = f"window_{idx:05d}.npy"
        feat_rel_path = f"features/Pham4/{feat_filename}"
        
        rows.append({
            "window_index": idx,
            "window_idx": idx,  # Included for backwards compatibility with dataset.py
            "feature_file_path": feat_rel_path,
            "gt_timestamp": center_time,
            "x": float(x),
            "y": float(y),
            "z": float(z),
        })

    # ── Step 4: Output CSV Metadata Index ──────────────────────────────────────
    print(f"\n[4/4] Generating synchronized master metadata index...")
    
    df = pd.DataFrame(rows)
    
    # Re-order columns strictly according to requirements with window_idx backup
    df = df[["window_index", "window_idx", "feature_file_path", "gt_timestamp", "x", "y", "z"]]
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    
    print(f"  SUCCESS: Aligned labels successfully saved -> {out_csv}")
    print(f"  SUCCESS: Synced {len(df)} feature windows to the trajectory timeline.")
    print("\nMetadata Preview:")
    print(df.head(5))


if __name__ == "__main__":
    main()