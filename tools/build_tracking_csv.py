"""
tools/build_tracking_csv.py — Offline CSV builder CLI
Usage:
    python tools/build_tracking_csv.py \
        --input  data/dummy_detections_200f.csv \
        --output results/sample_tracking_output_200f.csv \
        --fps 25.0
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.tracking_csv_builder import TrackingCSVBuilder
from src.pipeline.pitch_mapper import PitchMapper


def main(args):
    input_path  = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}"); sys.exit(1)

    pitch_mapper = None
    if args.homography:
        h = Path(args.homography)
        if h.exists():
            cfg = json.loads(h.read_text())
            pitch_mapper = PitchMapper(src_points=cfg["src_points"], dst_points=cfg["dst_points"])
            print(f"[INFO] Loaded homography from {h}")

    builder = TrackingCSVBuilder(pitch_mapper=pitch_mapper, fps=args.fps, ema_alpha=args.ema_alpha)
    print(f"[INFO] Loading detections from {input_path} ...")
    builder.load_from_csv(str(input_path))
    print("[INFO] Computing features and writing CSV ...")
    builder.finalize_and_write(str(output_path))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--homography", default=None)
    p.add_argument("--fps",        type=float, default=30.0)
    p.add_argument("--ema_alpha",  type=float, default=0.35)
    main(p.parse_args())
