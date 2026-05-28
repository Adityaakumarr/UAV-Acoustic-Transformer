"""
post_train.py
─────────────
Automated post-training pipeline. Runs evaluation, full inference,
and visualization after model training completes.

Usage:
    python post_train.py
"""

import subprocess
import sys
import os

def run_step(name, cmd):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if result.returncode != 0:
        print(f"  [FAILED] {name} exited with code {result.returncode}")
        return False
    print(f"  [SUCCESS] {name}")
    return True


def main():
    print("=" * 60)
    print("  POST-TRAINING PIPELINE")
    print("  Running evaluation, inference, and visualization...")
    print("=" * 60)

    # Check that the trained model exists
    if not os.path.exists("checkpoints/best_model.pth"):
        print("[ERROR] No trained model found at checkpoints/best_model.pth")
        print("        Please ensure training has completed first.")
        return

    steps = [
        ("1/3 - Evaluate Model (APE, Dx, Dy, Dz)", ["evaluate.py"]),
        ("2/3 - Full Trajectory Inference (1877 frames)", ["inference_full.py"]),
        ("3/3 - Generate Publication 3D Plots", ["visualize_3d_pham4_paper.py"]),
    ]

    results = []
    for name, cmd in steps:
        ok = run_step(name, cmd)
        results.append((name, ok))

    # Summary
    print(f"\n{'='*60}")
    print("  POST-TRAINING PIPELINE COMPLETE")
    print(f"{'='*60}")
    for name, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {name}")

    print(f"\nOutputs saved to:")
    print(f"  - outputs/metrics.csv")
    print(f"  - outputs/predictions.csv")
    print(f"  - outputs/predictions_full.csv")
    print(f"  - outputs/trajectory_3d_pham4_final.png")
    print(f"  - outputs/trajectory_3d_interact_pham4.html")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
