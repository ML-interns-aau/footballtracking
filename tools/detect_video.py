"""
tools/detect_video.py

Run YOLO detection on a video and write an annotated MP4.

Usage:
    python tools/detect_video.py --input path/to/video.mp4 --output detected.mp4
"""

import cv2
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.detector import FootballDetector


def try_get_attr(obj, *names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def main(args):
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = args.fps or (cap.get(cv2.CAP_PROP_FPS) or 25.0)

    out_path = Path(args.output)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    detector = FootballDetector(model_path=args.model_path, conf=args.conf, iou=args.iou, device=args.device)

    frame_idx = 0
    print(f"[INFO] Running detection on {args.input}, writing to {out_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        detections = detector.detect(frame)

        # Draw detections (use xyxy and class_id arrays)
        xyxy = try_get_attr(detections, 'xyxy', 'boxes', default=None)
        class_ids = try_get_attr(detections, 'class_id', 'class_ids', default=None)
        confs = try_get_attr(detections, 'confidence', 'conf', 'confidence_scores', default=None)

        if xyxy is not None and len(xyxy):
            for i, box in enumerate(xyxy):
                try:
                    x1, y1, x2, y2 = map(int, box[:4])
                except Exception:
                    continue
                cid = int(class_ids[i]) if class_ids is not None else -1
                label = detector.CLASS_NAMES_DICT.get(cid, str(cid)) if hasattr(detector, 'CLASS_NAMES_DICT') else str(cid)
                conf = (confs[i] if confs is not None and i < len(confs) else None)
                text = f"{label}" + (f" {conf:.2f}" if conf is not None else "")

                color = (0, 200, 0) if cid == 0 else (0, 0, 200)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[DONE] Wrote {frame_idx} frames to {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', default='results/detected.mp4')
    p.add_argument('--model_path', default='yolov8m_fixed.pt')
    p.add_argument('--conf', type=float, default=0.30)
    p.add_argument('--iou', type=float, default=0.40)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--max_frames', type=int, default=0)
    p.add_argument('--fps', type=float, default=0)
    args = p.parse_args()
    main(args)
