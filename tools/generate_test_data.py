"""Generate synthetic detection data for unit/integration testing.

Produces a CSV of dummy bounding-box detections covering 6 objects
(4 players across 2 teams, 1 referee, 1 ball) over N frames.

Usage:
    python tools/generate_test_data.py
    python tools/generate_test_data.py --frames 500 --output data/my_detections.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Synthetic objects: [team_id, player_id, x0, y0, vx, vy]
# ---------------------------------------------------------------------------

_OBJECTS: list[list[float]] = [
    [0,   1,  100, 200,  2.0,  0.5],
    [0,   2,  300, 180,  1.5,  0.2],
    [1,   3,  800, 190,  1.0,  0.3],
    [1,   4,  950, 210,  0.8, -0.1],
    [2,   5,  530, 300,  0.5,  0.1],   # team_id=2 → referee
    [32, 99,  520, 380,  5.0,  2.0],   # class_id=32 → ball
]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def generate_data(num_frames: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for frame in range(1, num_frames + 1):
        for obj in _OBJECTS:
            team_id, player_id, x, y, vx, vy = obj
            nx = x + vx * (frame - 1) + rng.normal(0, 0.5)
            ny = y + vy * (frame - 1) + rng.normal(0, 0.5)
            records.append({
                "frame":      frame,
                "team_id":    int(team_id),
                "player_id":  int(player_id),
                "bb_left":    round(nx, 2),
                "bb_top":     round(ny, 2),
                "bb_width":   60  if team_id != 32 else 22,
                "bb_height":  120 if team_id != 32 else 22,
                "confidence": round(0.9 + rng.uniform(0, 0.09), 4),
                "extra":      -1,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic detection CSV")
    parser.add_argument("--frames", type=int,  default=200,
                        help="Number of frames to generate (default: 200)")
    parser.add_argument("--output", type=str,  default=None,
                        help="Output CSV path (default: <project_root>/data/dummy_detections_200f.csv)")
    parser.add_argument("--seed",   type=int,  default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    out_path = Path(args.output) if args.output else project_root / "data" / "dummy_detections_200f.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_data(num_frames=args.frames, seed=args.seed)
    df.to_csv(out_path, index=False)
    print(f"[OK] Generated {args.frames} frames ({len(df)} rows) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
