"""
Quick verification for the SoccerNet -> YOLO conversion.

Usage:
    python scripts/verify_conversion.py
    python scripts/verify_conversion.py --label-dir datasets/soccernet_yolo/labels/train
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


CLASS_NAMES = [
    "player_left",
    "player_right",
    "goalkeeper_left",
    "goalkeeper_right",
    "referee",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify YOLO labels after conversion")
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("datasets/soccernet_yolo/labels/train"),
        help="YOLO label directory to inspect",
    )
    args = parser.parse_args()

    if not args.label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {args.label_dir}")

    counts = Counter()
    n_files = 0
    for label_file in args.label_dir.glob("*.txt"):
        n_files += 1
        text = label_file.read_text().strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            class_id = int(parts[0])
            counts[class_id] += 1

    print(f"Label files: {n_files}")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  class {class_id} ({class_name}): {counts[class_id]:,}")


if __name__ == "__main__":
    main()
