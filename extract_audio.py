"""
extract_audio.py
────────────────
Extract 4-channel audio from MMAUD ROS1 bag files.

Output structure:
audio/
└── <sequence_name>/
    ├── ch1.wav
    ├── ch2.wav
    ├── ch3.wav
    └── ch4.wav

Usage
-----
python extract_audio.py

Optional:
python extract_audio.py --bag_dir dataset/bags --out_dir audio

Requirements
------------
pip install rosbags soundfile numpy tqdm
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SR = 41800

AUDIO_TOPICS = [
    "/audio1/audio",
    "/audio2/audio",
    "/audio3/audio",
    "/audio4/audio",
]

TOPIC_TO_CHANNEL = {
    "/audio1/audio": 1,
    "/audio2/audio": 2,
    "/audio3/audio": 3,
    "/audio4/audio": 4,
}


# ─────────────────────────────────────────────────────────────────────────────
# Extract audio from one bag
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio_from_bag(
    bag_path: Path,
    out_dir: Path,
    sample_rate: int = DEFAULT_SR,
):

    try:
        from rosbags.rosbag1 import Reader
    except ImportError:
        print("\nERROR: rosbags not installed.")
        print("Run:")
        print("    pip install rosbags")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    buffers = defaultdict(list)

    print(f"\nProcessing: {bag_path.name}")

    with Reader(str(bag_path)) as reader:

        available_topics = [
            conn.topic
            for conn in reader.connections
        ]

        audio_topics_found = [
            t for t in AUDIO_TOPICS
            if t in available_topics
        ]

        print(f"  Found audio topics: {audio_topics_found}")

        if not audio_topics_found:
            print("  No audio topics found.")
            return False

        connections = [
            c for c in reader.connections
            if c.topic in AUDIO_TOPICS
        ]

        for conn, timestamp, rawdata in reader.messages(
            connections=connections
        ):

            try:
                # audio_common_msgs/AudioData format:
                # first 4 bytes = array length
                # remaining bytes = PCM int16 samples

                if len(rawdata) <= 4:
                    continue

                pcm_bytes = rawdata[4:]

                samples = (
                    np.frombuffer(
                        pcm_bytes,
                        dtype=np.int16
                    ).astype(np.float32)
                    / 32768.0
                )

                ch = TOPIC_TO_CHANNEL[conn.topic]

                buffers[ch].append(samples)

            except Exception as e:
                print(f"  WARNING: parse error on {conn.topic}: {e}")

    if not buffers:
        print("  ERROR: no audio extracted.")
        return False

    # Save WAV files
    for ch in range(1, 5):

        if ch not in buffers:
            print(f"  WARNING: no data for channel {ch}")
            continue

        audio = np.concatenate(buffers[ch])

        out_path = out_dir / f"ch{ch}.wav"

        sf.write(
            str(out_path),
            audio,
            sample_rate
        )

        duration = len(audio) / sample_rate

        print(
            f"  Saved: {out_path} "
            f"({duration:.2f} s)"
        )

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Demo generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_demo_audio(
    out_dir: Path,
    duration: float = 10.0,
    sample_rate: int = DEFAULT_SR,
):

    out_dir.mkdir(parents=True, exist_ok=True)

    t = np.linspace(
        0,
        duration,
        int(duration * sample_rate),
        endpoint=False
    )

    freqs = [100, 200, 400, 800]

    base_signal = sum(
        0.25 * np.sin(2 * np.pi * f * t)
        for f in freqs
    )

    delays_us = [0, 50, 120, 200]

    for ch, delay_us in enumerate(delays_us, start=1):

        delay_samples = int(
            delay_us * 1e-6 * sample_rate
        )

        signal = np.roll(
            base_signal,
            delay_samples
        )

        signal += 0.02 * np.random.randn(len(signal))

        sf.write(
            str(out_dir / f"ch{ch}.wav"),
            signal.astype(np.float32),
            sample_rate
        )

    print(f"Demo audio saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():

    parser = argparse.ArgumentParser(
        description="Extract 4-channel audio from MMAUD ROS bags"
    )

    parser.add_argument(
        "--bag_dir",
        type=str,
        default="dataset/bags",
        help="Directory containing .bag files"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="audio",
        help="Output directory"
    )

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=DEFAULT_SR,
        help="Audio sample rate"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate synthetic demo audio"
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():

    args = parse_args()

    out_root = Path(args.out_dir)

    if args.demo:

        generate_demo_audio(
            out_root / "demo",
            sample_rate=args.sample_rate
        )

        return

    bag_dir = Path(args.bag_dir)

    if not bag_dir.exists():
        print(f"Bag directory not found: {bag_dir}")
        return

    bag_files = sorted(
        bag_dir.glob("*.bag")
    )

    if not bag_files:
        print(f"No .bag files found in {bag_dir}")
        return

    print(f"Found {len(bag_files)} bag file(s).")

    for bag_path in tqdm(
        bag_files,
        desc="Extracting bags"
    ):

        seq_name = bag_path.stem

        out_seq_dir = out_root / seq_name

        try:
            extract_audio_from_bag(
                bag_path,
                out_seq_dir,
                args.sample_rate
            )

        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nExtraction complete.")

    print(
        "\nOutput structure:\n"
        "audio/<sequence>/ch1.wav ... ch4.wav"
    )


if __name__ == "__main__":
    main()