"""
extract_pham4.py
────────────────
Extract 4-channel audio and 3D pose ground truth from a MMAUD ROS1 bag file.

Outputs
───────
  audio/Pham4/ch1.wav … ch4.wav
  features/Pham4/labels.csv

Usage
─────
    python extract_pham4.py --bag Pham4.bag
    python extract_pham4.py --bag Pham4.bag --list_topics   # inspect bag first
    python extract_pham4.py --bag Pham4.bag --pose_topic /mavros/local_position/pose

Requirements
────────────
    pip install rosbags soundfile pandas numpy
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SR = 41800

AUDIO_KEYWORDS = ["audio", "mic", "microphone", "sound"]
POSE_KEYWORDS  = ["pose", "position", "odom", "odometry", "ground_truth", "mavros"]

# Hard-coded per-microphone topics used in the MMAUD / Pham4 bag
AUDIO_TOPICS = [
    "/audio1/audio",
    "/audio2/audio",
    "/audio3/audio",
    "/audio4/audio",
]
TOPIC_TO_CHANNEL = {t: i + 1 for i, t in enumerate(AUDIO_TOPICS)}


# ──────────────────────────────────────────────────────────────────────────────
# ROS bag helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_reader():
    """Import rosbags Reader, or exit with a clear message."""
    try:
        from rosbags.rosbag1 import Reader
        return Reader
    except ImportError:
        print("\nERROR: rosbags is not installed.")
        print("Install it with:  pip install rosbags")
        sys.exit(1)


def deserialize_message(rawdata: bytes, msgtype: str):
    """
    Deserialize a raw ROS 1 message payload using rosbags.

    rosbags stores ROS 1 messages in their native on-wire format.
    We convert to CDR (ROS 2 common format) then deserialize.
    """
    try:
        from rosbags.serde import deserialize_cdr, ros1_to_cdr
        return deserialize_cdr(ros1_to_cdr(rawdata, msgtype), msgtype)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to deserialize message of type '{msgtype}': {exc}\n"
            "Make sure rosbags >= 0.9.11 is installed:  pip install rosbags"
        ) from exc


# ──────────────────────────────────────────────────────────────────────────────
# Step 0 — List topics
# ──────────────────────────────────────────────────────────────────────────────

def list_topics(bag_path: str) -> list[str]:
    Reader = _get_reader()
    print("\n========== TOPICS ==========\n")
    seen = {}
    with Reader(bag_path) as reader:
        for conn in reader.connections:
            if conn.topic not in seen:
                seen[conn.topic] = conn.msgtype
                print(f"  {conn.topic:<55} {conn.msgtype}")
    return list(seen.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Step 0b — Auto-detect audio and pose topics
# ──────────────────────────────────────────────────────────────────────────────

def detect_topics(topics: list[str]) -> tuple[str | None, str | None]:
    audio_topic = next(
        (t for t in topics if any(k in t.lower() for k in AUDIO_KEYWORDS)),
        None,
    )
    pose_topic = next(
        (t for t in topics if any(k in t.lower() for k in POSE_KEYWORDS)),
        None,
    )
    return audio_topic, pose_topic


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Extract 4-channel audio
# ──────────────────────────────────────────────────────────────────────────────

def extract_audio(bag_path: str, out_dir: Path, sample_rate: int = DEFAULT_SR) -> bool:
    """
    Read audio from the four per-microphone topics (/audio1/audio … /audio4/audio)
    and write ch1.wav … ch4.wav into out_dir.

    audio_common_msgs/AudioData wire layout (ROS 1)
    ─────────────────────────────────────────────────
      bytes 0-3  : uint32  array length   (little-endian)
      bytes 4+   : int16[] PCM samples    (little-endian)

    We skip the 4-byte length header and cast the remaining bytes to int16.
    """
    Reader = _get_reader()
    out_dir.mkdir(parents=True, exist_ok=True)

    buffers: dict[int, list[np.ndarray]] = defaultdict(list)

    print(f"\n[1/3] Extracting audio from: {bag_path}")

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic in AUDIO_TOPICS]

        if not connections:
            print("  WARNING: none of the expected audio topics found in this bag.")
            print("  Expected:", AUDIO_TOPICS)
            print("  Run with --list_topics to see what's available.")
            return False

        for conn, _timestamp, rawdata in reader.messages(connections=connections):
            try:
                # Skip the 4-byte ROS array-length prefix
                pcm_bytes = rawdata[4:]
                samples = (
                    np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            except Exception as exc:
                print(f"  WARNING: audio parse error on {conn.topic}: {exc}")
                continue

            ch = TOPIC_TO_CHANNEL[conn.topic]
            buffers[ch].append(samples)

    if not buffers:
        print("  ERROR: no audio data extracted.")
        return False

    for ch in range(1, 5):
        frames = buffers.get(ch)
        if not frames:
            print(f"  WARNING: no data for channel {ch}")
            continue
        audio = np.concatenate(frames)
        wav_path = out_dir / f"ch{ch}.wav"
        sf.write(str(wav_path), audio, sample_rate)
        duration = len(audio) / sample_rate
        print(f"  Saved: {wav_path}  ({duration:.2f} s,  {len(audio):,} samples)")

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Extract pose ground truth
# ──────────────────────────────────────────────────────────────────────────────

def extract_poses(bag_path: str, pose_topic: str, out_csv: Path) -> bool:
    """
    Read geometry_msgs/PoseStamped (or nav_msgs/Odometry) messages and write
    a CSV with columns: timestamp, x, y, z.
    """
    Reader = _get_reader()

    print(f"\n[2/3] Extracting poses from topic: {pose_topic}")

    rows = []

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == pose_topic]

        if not connections:
            print(f"  ERROR: topic '{pose_topic}' not found in bag.")
            return False

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            try:
                msg = deserialize_message(rawdata, conn.msgtype)
            except Exception as exc:
                print(f"  WARNING: deserialization failed: {exc}")
                continue

            x = y = z = None
            try:
                # geometry_msgs/PoseStamped  →  msg.pose.position
                if hasattr(msg, "pose") and hasattr(msg.pose, "position"):
                    pos = msg.pose.position
                    x, y, z = float(pos.x), float(pos.y), float(pos.z)

                # nav_msgs/Odometry  →  msg.pose.pose.position
                elif hasattr(msg, "pose") and hasattr(msg.pose, "pose"):
                    pos = msg.pose.pose.position
                    x, y, z = float(pos.x), float(pos.y), float(pos.z)

                # geometry_msgs/PointStamped  →  msg.point
                elif hasattr(msg, "point"):
                    pos = msg.point
                    x, y, z = float(pos.x), float(pos.y), float(pos.z)

            except Exception as exc:
                print(f"  WARNING: could not read position fields: {exc}")
                continue

            if x is not None:
                rows.append({
                    "timestamp": timestamp * 1e-9,   # nanoseconds → seconds
                    "x": x,
                    "y": y,
                    "z": z,
                })

    if not rows:
        print("  ERROR: no pose data extracted.")
        return False

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}  ({len(df):,} poses)")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Align poses → per-window labels
# ──────────────────────────────────────────────────────────────────────────────

def align_labels(
    poses_csv: Path,
    audio_dir: Path,
    out_csv: Path,
    window_ms: float = 400.0,
    overlap: float = 0.75,
) -> None:
    """
    For each sliding-window of audio, find the nearest pose timestamp and
    write labels.csv with columns: window_idx, x, y, z.
    """
    print("\n[3/3] Aligning window labels to poses …")

    df = pd.read_csv(poses_csv)

    wav_path = audio_dir / "ch1.wav"
    if not wav_path.exists():
        raise FileNotFoundError(f"Reference WAV not found: {wav_path}")

    info = sf.info(str(wav_path))
    sr       = info.samplerate
    n_samples = info.frames

    win_samples = int(window_ms * 1e-3 * sr)
    hop_samples = int(win_samples * (1 - overlap))

    rows = []
    idx  = 0
    start = 0

    while start + win_samples <= n_samples:
        center_t = (start + win_samples / 2) / sr
        nearest  = (df["timestamp"] - center_t).abs().idxmin()
        pose     = df.iloc[nearest]
        rows.append({
            "window_idx": idx,
            "x": pose["x"],
            "y": pose["y"],
            "z": pose["z"],
        })
        idx   += 1
        start += hop_samples

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}  ({len(rows):,} windows)")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract audio + pose labels from a MMAUD ROS1 bag file."
    )
    p.add_argument("--bag",         type=str, default="Pham4.bag",
                   help="Path to the .bag file")
    p.add_argument("--audio_topic", type=str, default=None,
                   help="Override auto-detected audio topic")
    p.add_argument("--pose_topic",  type=str, default=None,
                   help="Override auto-detected pose topic")
    p.add_argument("--out_audio",   type=str, default="audio/Pham4",
                   help="Output directory for WAV files")
    p.add_argument("--out_features",type=str, default="features/Pham4",
                   help="Output directory for labels.csv (and later .npy features)")
    p.add_argument("--list_topics", action="store_true",
                   help="Print all topics in the bag and exit")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    bag_path = str(Path(args.bag))

    if not Path(bag_path).exists():
        print(f"\nERROR: bag file not found: {bag_path}")
        return

    print("\n========== PHAM4 EXTRACTOR ==========")

    # Inspect topics
    topics = list_topics(bag_path)
    if args.list_topics:
        return

    # Resolve topics
    auto_audio, auto_pose = detect_topics(topics)
    audio_topic = args.audio_topic or auto_audio
    pose_topic  = args.pose_topic  or auto_pose

    print("\nTopics selected:")
    print(f"  Audio : {audio_topic}")
    print(f"  Pose  : {pose_topic}")

    if audio_topic is None:
        print("\nERROR: no audio topic found. Use --list_topics to inspect the bag,")
        print("       then re-run with  --audio_topic <topic>.")
        return

    audio_dir   = Path(args.out_audio)
    feat_dir    = Path(args.out_features)
    poses_csv   = feat_dir / "poses_raw.csv"
    labels_csv  = feat_dir / "labels.csv"

    # ── Step 1: audio ─────────────────────────────────────────────────────
    ok = extract_audio(bag_path, audio_dir)
    if not ok:
        return

    # ── Step 2: poses ─────────────────────────────────────────────────────
    if pose_topic is not None:
        extract_poses(bag_path, pose_topic, poses_csv)
    else:
        print("\nWARNING: no pose topic found — skipping ground-truth extraction.")
        print("         Use --pose_topic <topic> if poses exist under a different name.")

    # ── Step 3: labels ────────────────────────────────────────────────────
    if poses_csv.exists():
        align_labels(poses_csv, audio_dir, labels_csv)
    else:
        print("\nINFO: poses_raw.csv not found — skipping label alignment.")

    print("\n========== DONE ==========")
    print(f"\nAudio  : {audio_dir}")
    print(f"Labels : {labels_csv}")
    print("\nNext steps:")
    print("  python feature_extraction.py")
    print("  python train.py")


if __name__ == "__main__":
    main()
