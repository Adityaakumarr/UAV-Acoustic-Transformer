"""
evaluate.py
───────────
Evaluate a trained AcousticTransformer on the test set.

Metrics (matching the paper)
─────────────────────────────
  Dx   — mean absolute error along X axis
  Dy   — mean absolute error along Y axis
  Dz   — mean absolute error along Z axis
  APE  — Absolute Position Error  (Euclidean distance, primary metric)

Usage
─────
    python evaluate.py                          # uses best_model from config
    python evaluate.py --ckpt path/to/model.pth
    python evaluate.py --synthetic              # evaluate on synthetic data
"""

import argparse
import yaml
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from model   import build_model
from dataset import build_dataloaders


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict, ckpt_path: str, device: torch.device):
    model = build_model(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device) -> dict:
    """
    Run inference over the loader and compute per-axis + APE metrics.

    Returns
    -------
    dict with keys: APE, Dx, Dy, Dz, predictions (np.ndarray), targets (np.ndarray)
    """
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for feat, hist, lbl in tqdm(loader, desc="Evaluating"):
            feat = feat.to(device, non_blocking=True)
            hist = hist.to(device, non_blocking=True)
            lbl  = lbl.to(device,  non_blocking=True)

            pred = model(feat, hist)    # (B, 3)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(lbl.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)   # (N, 3)
    targets = np.concatenate(all_targets, axis=0)   # (N, 3)

    errors   = np.abs(preds - targets)               # (N, 3)
    ape      = np.linalg.norm(preds - targets, axis=1).mean()

    metrics = {
        "APE": float(ape),
        "Dx":  float(errors[:, 0].mean()),
        "Dy":  float(errors[:, 1].mean()),
        "Dz":  float(errors[:, 2].mean()),
        "predictions": preds,
        "targets":     targets,
    }
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print & save results
# ──────────────────────────────────────────────────────────────────────────────

_METHODS_TABLE = {
    "AudioNet":       2.80,
    "DroneChase":     2.64,
    "TAME":           0.55,
}

def print_results(metrics: dict):
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  APE  : {metrics['APE']:.4f} m")
    print(f"  Dx   : {metrics['Dx']:.4f} m")
    print(f"  Dy   : {metrics['Dy']:.4f} m")
    print(f"  Dz   : {metrics['Dz']:.4f} m")
    print("=" * 50)

    print("\n-- Comparison with paper baselines (APE) --")
    rows = list(_METHODS_TABLE.items()) + [("Ours", metrics["APE"])]
    for name, val in rows:
        marker = " [OURS]" if name == "Ours" else ""
        print(f"  {name:<16} {val:.2f}{marker}")
    print()


def save_results(metrics: dict, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Scalar metrics CSV
    scalar_path = str(Path(out_dir) / "metrics.csv")
    df_metrics  = pd.DataFrame([{
        "APE": metrics["APE"],
        "Dx":  metrics["Dx"],
        "Dy":  metrics["Dy"],
        "Dz":  metrics["Dz"],
    }])
    df_metrics.to_csv(scalar_path, index=False)
    print(f"Metrics saved -> {scalar_path}")

    # Prediction / target CSV
    pred_path = str(Path(out_dir) / "predictions.csv")
    preds   = metrics["predictions"]
    targets = metrics["targets"]
    df_pred = pd.DataFrame({
        "pred_x": preds[:, 0],   "pred_y": preds[:, 1],   "pred_z": preds[:, 2],
        "true_x": targets[:, 0], "true_y": targets[:, 1], "true_z": targets[:, 2],
        "APE":    np.linalg.norm(preds - targets, axis=1),
    })
    df_pred.to_csv(pred_path, index=False)
    print(f"Predictions saved -> {pred_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AcousticTransformer")
    p.add_argument("--config",    type=str, default="config.yaml")
    p.add_argument("--ckpt",      type=str, default=None,
                   help="Checkpoint path (default: best_model from config)")
    p.add_argument("--synthetic", action="store_true",
                   help="Evaluate on synthetic data (no dataset needed)")
    p.add_argument("--out_dir",   type=str, default=None,
                   help="Directory to save results (default: outputs from config)")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = args.ckpt or cfg["paths"]["best_model"]
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Train first:  python train.py  (or --synthetic for a quick test)")
        return

    model = load_model(cfg, ckpt_path, device)
    print(f"Loaded checkpoint: {ckpt_path}")

    _, _, test_loader, _ = build_dataloaders(cfg, synthetic=args.synthetic)

    metrics = evaluate(model, test_loader, device)
    print_results(metrics)

    out_dir = args.out_dir or cfg["paths"]["outputs_dir"]
    save_results(metrics, out_dir)


if __name__ == "__main__":
    main()
