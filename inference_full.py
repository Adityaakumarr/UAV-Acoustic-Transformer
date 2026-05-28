"""
inference_full.py
─────────────────
Inference script to run the trained AcousticTransformer model over the ENTIRE
aligned dataset sequence to obtain predictions for the complete 4-loop flight trajectory.
Saves predictions to outputs/predictions_full.csv.
"""

import os
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from model   import build_model
from dataset import UAVTrajectoryDataset, FeatureNormalizer


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load trained best model checkpoint
    ckpt_path = cfg["paths"]["best_model"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_path}")

    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded best model checkpoint from {ckpt_path}")

    # 2. Load persistent normalizer
    norm_path = os.path.join(cfg["paths"]["checkpoints_dir"], "normalizer.pt")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Normalizer not found at: {norm_path}")

    normalizer = FeatureNormalizer()
    normalizer.load(norm_path)
    print(f"Loaded normalizer from {norm_path}")

    # 3. Load entire dataset sequence
    features_dir = cfg["paths"]["features_dir"]
    K = cfg["model"]["traj_seq_len"]

    print("Loading full sequence dataset...")
    full_ds = UAVTrajectoryDataset(
        features_dir=features_dir,
        traj_seq_len=K,
        normalizer=normalizer,
        augment=False
    )

    print(f"Total full sequence samples to predict: {len(full_ds)}")

    # High-speed batch inference using DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(
        full_ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,  # Safe for Windows
        pin_memory=True
    )

    all_preds = []
    all_targets = []

    # 4. Predict entire trajectory in optimized batches
    with torch.no_grad():
        for feat, hist, lbl in tqdm(loader, desc="Predicting Full Trajectory"):
            feat = feat.to(device)
            hist = hist.to(device)

            pred = model(feat, hist)  # (B, 3)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(lbl.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Use actual model predictions directly
    pred_x_out = preds[:, 0]
    pred_y_out = preds[:, 1]
    pred_z_out = preds[:, 2]

    # 5. Save complete coordinates to outputs/predictions_full.csv
    out_dir = cfg["paths"]["outputs_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "predictions_full.csv")

    df_full = pd.DataFrame({
        "pred_x": pred_x_out,   "pred_y": pred_y_out,   "pred_z": pred_z_out,
        "true_x": targets[:, 0], "true_y": targets[:, 1], "true_z": targets[:, 2],
        "APE":    np.linalg.norm(preds - targets, axis=1),
    })

    df_full.to_csv(out_csv, index=False)
    print(f"SUCCESS: Saved full predictions to '{out_csv}' (total {len(df_full)} frames)")


if __name__ == "__main__":
    main()
