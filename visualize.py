"""
visualize.py
────────────
Visualise predicted vs ground-truth UAV trajectories in 3D.

Generates
─────────
  trajectory_3d.png        — 3-D perspective plot
  trajectory_axes.png      — X / Y / Z vs time panels
  error_histogram.png      — APE distribution

Usage
─────
    python visualize.py --pred outputs/predicted_trajectory.csv \
                        --gt   outputs/predictions.csv          # evaluate.py output

    python visualize.py --demo  # generate random demo data and visualise
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # noqa: F401 (needed for 3d projection)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting functions
# ──────────────────────────────────────────────────────────────────────────────

def plot_3d_trajectory(pred: np.ndarray, gt: np.ndarray, out_path: str):
    """
    3-D side-by-side: ground truth (blue) vs prediction (red).
    """
    fig = plt.figure(figsize=(13, 6))

    for col_idx, (positions, title, color) in enumerate([
        (gt,   "Ground Truth", "royalblue"),
        (pred, "Predicted",    "crimson"),
    ]):
        ax = fig.add_subplot(1, 2, col_idx + 1, projection="3d")
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        n = len(x)
        c = plt.cm.viridis(np.linspace(0, 1, n))

        for i in range(n - 1):
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=c[i], linewidth=1.5)

        ax.scatter(*positions[0],  s=60, c="green", zorder=5, label="Start")
        ax.scatter(*positions[-1], s=60, c="red",   zorder=5, label="End")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(fontsize=8)

    plt.suptitle("UAV 3D Trajectory Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"3D plot saved -> {out_path}")


def plot_axes_vs_time(pred: np.ndarray, gt: np.ndarray, out_path: str):
    """
    Three-row panel: X, Y, Z coordinate over time for GT and prediction.
    """
    axes_labels = ["X (m)", "Y (m)", "Z (m)"]
    t = np.arange(len(pred))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, (ax, lbl) in enumerate(zip(axes, axes_labels)):
        ax.plot(t, gt[:, i],   label="Ground Truth", color="royalblue", linewidth=1.5)
        ax.plot(t, pred[:, i], label="Predicted",    color="crimson",   linewidth=1.5,
                linestyle="--")
        ax.set_ylabel(lbl)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Window index")
    fig.suptitle("X / Y / Z over Time", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Axes plot saved -> {out_path}")


def plot_ape_histogram(pred: np.ndarray, gt: np.ndarray, out_path: str):
    """APE per-window histogram with mean/median lines."""
    ape = np.linalg.norm(pred - gt, axis=1)
    mean_ape   = ape.mean()
    median_ape = np.median(ape)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ape, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(mean_ape,   color="red",    linestyle="--", linewidth=1.5,
               label=f"Mean  APE = {mean_ape:.3f} m")
    ax.axvline(median_ape, color="orange", linestyle=":",  linewidth=1.5,
               label=f"Median APE = {median_ape:.3f} m")
    ax.set_xlabel("APE (m)")
    ax.set_ylabel("Count")
    ax.set_title("Absolute Position Error Distribution", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Histogram saved -> {out_path}")


def plot_combined_3d(pred: np.ndarray, gt: np.ndarray, out_path: str):
    """Overlay both trajectories in one 3-D plot with error segments."""
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # Ground truth
    ax.plot(gt[:, 0],   gt[:, 1],   gt[:, 2],   color="royalblue", linewidth=2,
            label="Ground Truth")
    # Prediction
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color="crimson",   linewidth=2,
            linestyle="--", label="Predicted")

    # Error segments (every Nth point for clarity)
    step = max(1, len(pred) // 40)
    for i in range(0, len(pred), step):
        ax.plot([gt[i, 0], pred[i, 0]],
                [gt[i, 1], pred[i, 1]],
                [gt[i, 2], pred[i, 2]],
                color="gray", linewidth=0.5, alpha=0.5)

    ax.scatter(*gt[0],   s=80, c="green", zorder=5, label="Start")
    ax.scatter(*gt[-1],  s=80, c="navy",  zorder=5, label="End (GT)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Ground Truth vs Predicted Trajectory\n(grey lines = position error)",
                 fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Combined 3D plot saved -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_evaluate_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load CSV produced by evaluate.py  (columns: pred_x/y/z, true_x/y/z).
    Returns (pred, gt) each shape (N, 3).
    """
    df   = pd.read_csv(path)
    pred = df[["pred_x", "pred_y", "pred_z"]].values.astype(np.float32)
    gt   = df[["true_x", "true_y", "true_z"]].values.astype(np.float32)
    return pred, gt


def load_pred_and_gt(pred_csv: str, gt_csv: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load prediction CSV (from inference.py) and a separate ground-truth CSV.
    GT CSV must have columns: x, y, z.
    """
    pred_df = pd.read_csv(pred_csv)
    gt_df   = pd.read_csv(gt_csv)

    pred = pred_df[["x", "y", "z"]].values.astype(np.float32)
    gt   = gt_df[["x",  "y", "z"]].values.astype(np.float32)

    # Align lengths
    n = min(len(pred), len(gt))
    return pred[:n], gt[:n]


def generate_demo_data(n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Smooth random walk + small Gaussian noise for quick demo."""
    gt   = np.zeros((n, 3), dtype=np.float32)
    vel  = np.random.randn(3).astype(np.float32) * 0.05
    for i in range(1, n):
        vel  = vel * 0.95 + np.random.randn(3).astype(np.float32) * 0.02
        gt[i] = gt[i - 1] + vel

    pred = gt + np.random.randn(*gt.shape).astype(np.float32) * 0.15
    return pred, gt


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualise UAV trajectory predictions")
    p.add_argument("--pred",    type=str, default=None,
                   help="Prediction CSV (inference.py output)  — x,y,z columns")
    p.add_argument("--gt",      type=str, default=None,
                   help="Ground-truth CSV  — x,y,z  OR  evaluate.py predictions.csv")
    p.add_argument("--eval_csv",type=str, default=None,
                   help="evaluate.py predictions.csv (has both pred and gt columns)")
    p.add_argument("--out_dir", type=str, default="outputs",
                   help="Directory to save plots")
    p.add_argument("--demo",    action="store_true",
                   help="Generate random demo trajectories and visualise")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    if args.demo:
        print("Generating demo trajectories ...")
        pred, gt = generate_demo_data()
    elif args.eval_csv:
        pred, gt = load_evaluate_csv(args.eval_csv)
    elif args.pred and args.gt:
        pred, gt = load_pred_and_gt(args.pred, args.gt)
    else:
        print("Provide one of:\n"
              "  --demo\n"
              "  --eval_csv outputs/predictions.csv\n"
              "  --pred outputs/predicted_trajectory.csv  --gt path/to/gt.csv")
        return

    print(f"Loaded {len(pred)} trajectory points.")

    # ── Generate all plots ─────────────────────────────────────────────────
    plot_combined_3d    (pred, gt, str(out_dir / "trajectory_3d.png"))
    plot_3d_trajectory  (pred, gt, str(out_dir / "trajectory_side_by_side.png"))
    plot_axes_vs_time   (pred, gt, str(out_dir / "trajectory_axes.png"))
    plot_ape_histogram  (pred, gt, str(out_dir / "error_histogram.png"))

    # Quick summary
    ape = np.linalg.norm(pred - gt, axis=1)
    print(f"\nAPE  mean={ape.mean():.4f}  std={ape.std():.4f}  "
          f"min={ape.min():.4f}  max={ape.max():.4f}  (metres)")


if __name__ == "__main__":
    main()
