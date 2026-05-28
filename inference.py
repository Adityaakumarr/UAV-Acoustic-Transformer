"""
inference.py
────────────
Run the trained AcousticTransformer on a new sequence of 4-channel WAV
files and produce:
  • predicted_trajectory.csv
  • trajectory_plot.png

Usage
─────
    python inference.py --audio_dir audio/my_sequence
    python inference.py --audio_dir audio/my_sequence --ckpt checkpoints/best_model.pth
    python inference.py --demo      # generate synthetic audio and run inference
"""

import argparse
import yaml
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")                        # headless-safe backend
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from model              import build_model
from feature_extraction import (
    load_config, sliding_windows, build_feature_tensor
)


# ──────────────────────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────────────────────

def load_model(cfg: dict, ckpt_path: str, device: torch.device):
    model = build_model(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Inference pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(model, audio_dir: Path, cfg: dict, device: torch.device) -> np.ndarray:
    """
    Loads 4-channel WAV, extracts features, predicts trajectory window-by-window.

    Returns predicted_positions: np.ndarray of shape (N, 3)
    """
    sr         = cfg["audio"]["sample_rate"]
    win_ms     = cfg["audio"]["window_ms"]
    ovlp       = cfg["audio"]["overlap"]
    K          = cfg["model"]["traj_seq_len"]

    win_samples = int(win_ms * 1e-3 * sr)
    hop_samples = int(win_samples * (1 - ovlp))

    # ── Load WAV files ────────────────────────────────────────────────────
    wav_paths = [audio_dir / f"ch{i}.wav" for i in range(1, 5)]
    missing   = [p for p in wav_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing WAV files: {[str(p) for p in missing]}")

    channels = []
    for wp in wav_paths:
        data, file_sr = sf.read(str(wp))
        if file_sr != sr:
            import librosa
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        channels.append(data.astype(np.float32))

    min_len  = min(len(c) for c in channels)
    channels = [c[:min_len] for c in channels]

    # ── Sliding windows ───────────────────────────────────────────────────
    windows = sliding_windows(channels, win_samples, hop_samples)
    print(f"  {len(windows)} windows to infer …")

    predicted_positions = []
    traj_history = np.zeros((K, 3), dtype=np.float32)  # initialise with zeros

    with torch.no_grad():
        for win_chs in tqdm(windows, desc="Inference"):
            feat = build_feature_tensor(win_chs, cfg["audio"], cfg["features"])
            feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)         # (1,6,F,T)
            hist_t = torch.from_numpy(traj_history).unsqueeze(0).to(device) # (1,K,3)

            pred = model(feat_t, hist_t).squeeze(0).cpu().numpy()            # (3,)
            predicted_positions.append(pred)

            # Roll history forward
            traj_history = np.roll(traj_history, shift=-1, axis=0)
            traj_history[-1] = pred

    return np.array(predicted_positions)   # (N, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Save outputs
# ──────────────────────────────────────────────────────────────────────────────

def save_trajectory_csv(positions: np.ndarray, out_path: str):
    df = pd.DataFrame(positions, columns=["x", "y", "z"])
    df.index.name = "window_idx"
    df.to_csv(out_path)
    print(f"Trajectory CSV → {out_path}")


def plot_trajectory(positions: np.ndarray, out_path: str):
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # Colour trajectory by time
    n   = len(x)
    c   = plt.cm.plasma(np.linspace(0, 1, n))

    for i in range(n - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=c[i], linewidth=1.5)

    ax.scatter(*positions[0],  s=80, color="green",  zorder=5, label="Start")
    ax.scatter(*positions[-1], s=80, color="red",    zorder=5, label="End")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Predicted UAV Trajectory", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Trajectory plot → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Demo helper
# ──────────────────────────────────────────────────────────────────────────────

def generate_demo_audio(out_dir: Path, duration: float = 10.0, sr: int = 41800):
    """Create synthetic 4-channel audio for a quick demo run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration, int(duration * sr))
    base = sum(0.25 * np.sin(2 * np.pi * f * t) for f in [100, 200, 400, 800])
    for ch in range(1, 5):
        sig = base + 0.02 * np.random.randn(len(base))
        sf.write(str(out_dir / f"ch{ch}.wav"), sig.astype(np.float32), sr)
    print(f"Demo audio generated in {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run inference with AcousticTransformer")
    p.add_argument("--config",    type=str, default="config.yaml")
    p.add_argument("--audio_dir", type=str, default=None,
                   help="Directory with ch1.wav … ch4.wav")
    p.add_argument("--ckpt",      type=str, default=None,
                   help="Checkpoint path (default: best_model from config)")
    p.add_argument("--out_dir",   type=str, default=None,
                   help="Output directory (default: outputs from config)")
    p.add_argument("--demo",      action="store_true",
                   help="Generate synthetic audio and run inference (no dataset needed)")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir or cfg["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Demo: generate audio and create a dummy checkpoint if needed
    if args.demo:
        audio_dir = Path(cfg["paths"]["audio_dir"]) / "demo_inference"
        generate_demo_audio(audio_dir, sr=cfg["audio"]["sample_rate"])
    else:
        if args.audio_dir is None:
            print("Provide --audio_dir or use --demo")
            return
        audio_dir = Path(args.audio_dir)
        if not audio_dir.exists():
            print(f"Audio directory not found: {audio_dir}")
            return

    ckpt_path = args.ckpt or cfg["paths"]["best_model"]
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Train first:  python train.py  (or --synthetic for a quick test)")
        return

    model     = load_model(cfg, ckpt_path, device)
    positions = run_inference(model, audio_dir, cfg, device)

    save_trajectory_csv(positions, str(out_dir / "predicted_trajectory.csv"))
    plot_trajectory(    positions, str(out_dir / "trajectory_plot.png"))

    print(f"\nInference done.  Predicted {len(positions)} positions.")


if __name__ == "__main__":
    main()
