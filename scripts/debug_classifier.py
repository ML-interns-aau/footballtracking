"""
Debug script: visualize jersey crops and dominant Lab colors on a video clip.
Usage: python scripts/debug_classifier.py --video path/to/clip.mp4 --frames 60
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.detector import FootballDetector
from src.pipeline.team_classifier import TeamClassifier, normalize_frame


def _lab_to_bgr(color_lab: np.ndarray) -> tuple[int, int, int]:
    patch = np.zeros((1, 1, 3), dtype=np.uint8)
    lab = np.clip(np.asarray(color_lab).astype(np.float32), 0, 255).astype(np.uint8)
    patch[0, 0] = lab
    bgr = cv2.cvtColor(patch, cv2.COLOR_Lab2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _make_text_tile(text: str, width: int = 160, height: int = 48) -> np.ndarray:
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(tile, text, (8, height // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return tile


def _resize_keep_aspect(image: np.ndarray, target_width: int) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((1, target_width, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    if w <= 0:
        return np.zeros((1, target_width, 3), dtype=np.uint8)
    scale = target_width / float(w)
    target_height = max(1, int(round(h * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _stack_tiles(tiles: list[np.ndarray], tile_width: int = 180, max_cols: int = 4) -> np.ndarray:
    if not tiles:
        return np.zeros((1, tile_width, 3), dtype=np.uint8)

    prepared = []
    for tile in tiles:
        prepared.append(_resize_keep_aspect(tile, tile_width))

    rows = []
    for start in range(0, len(prepared), max_cols):
        row_tiles = prepared[start:start + max_cols]
        max_h = max(tile.shape[0] for tile in row_tiles)
        padded = []
        for tile in row_tiles:
            if tile.shape[0] < max_h:
                pad = np.zeros((max_h - tile.shape[0], tile.shape[1], 3), dtype=np.uint8)
                tile = np.vstack([tile, pad])
            padded.append(tile)
        rows.append(cv2.hconcat(padded))

    max_w = max(row.shape[1] for row in rows)
    normalized_rows = []
    for row in rows:
        if row.shape[1] < max_w:
            pad = np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8)
            row = np.hstack([row, pad])
        normalized_rows.append(row)

    return cv2.vconcat(normalized_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input clip")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to inspect")
    parser.add_argument("--model_path", default="yolov8m_fixed.pt")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.40)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    detector = FootballDetector(model_path=args.model_path, conf=args.conf, iou=args.iou, device=args.device)
    classifier = TeamClassifier()

    frame_idx = 0
    while frame_idx < args.frames:
        ret, frame = cap.read()
        if not ret:
            break

        norm = normalize_frame(frame)
        detections = detector.detect(frame)
        team_ids = classifier.assign_teams(norm, detections)

        print(f"Frame {frame_idx}: players={int((detections.class_id == 0).sum()) if detections.class_id is not None else 0}")

        crop_tiles: list[np.ndarray] = []
        center_tiles: list[np.ndarray] = []
        if detections.class_id is not None:
            for i, (bbox, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
                if int(class_id) != 0:
                    continue
                crop = classifier._get_jersey_crop(norm, bbox)
                if crop is None:
                    continue
                dom = classifier._extract_dominant_lab(crop)
                if dom is None:
                    continue
                team = int(team_ids[i]) if i < len(team_ids) else -1
                print(f"  player {i}: team={team} lab={np.array2string(dom, precision=2)}")
                print(f"    crop_shape={crop.shape}")

                crop_vis = crop.copy()
                cv2.rectangle(crop_vis, (0, 0), (crop_vis.shape[1] - 1, crop_vis.shape[0] - 1), (255, 255, 255), 1)
                cv2.putText(crop_vis, f"id {i} t {team}", (4, min(18, crop_vis.shape[0] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                crop_tiles.append(crop_vis)

        if classifier.is_fitted:
            labels = ["team_left", "team_right"]
            for idx, center in enumerate(classifier.kmeans.cluster_centers_[:2]):
                bgr = _lab_to_bgr(center)
                tile = np.full((50, 180, 3), bgr, dtype=np.uint8)
                cv2.putText(tile, f"{labels[idx]} Lab={np.array2string(center, precision=1)}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
                center_tiles.append(tile)

        top = cv2.hconcat([
            _resize_keep_aspect(frame, 640),
            _resize_keep_aspect(norm, 640),
        ])
        top_label = cv2.hconcat([
            _make_text_tile("original"),
            _make_text_tile("normalized"),
        ])
        crop_grid = _stack_tiles(crop_tiles, tile_width=180, max_cols=4)
        if center_tiles:
            center_grid = _stack_tiles(center_tiles, tile_width=220, max_cols=2)
        else:
            center_grid = _make_text_tile("cluster centers unavailable yet", width=360, height=48)

        canvas = cv2.vconcat([
            top_label,
            top,
            _make_text_tile("jersey crops / dominant Lab"),
            crop_grid,
            _make_text_tile("cluster centers"),
            center_grid,
        ])

        cv2.imshow("debug_classifier", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
