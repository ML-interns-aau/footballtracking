"""
Converts SoccerNet-GSR Labels-GameState.json annotations to YOLO format.

For each clip, reads the JSON, maps role+team -> class_id, converts COCO-style
bboxes to YOLO normalized format, and writes one .txt label file per image frame.
Images are symlinked (not copied) to save disk space.

Usage:
    python scripts/convert_annotations.py
    python scripts/convert_annotations.py --data-dir data/SoccerNetGS --out-dir datasets/soccernet_yolo
    python scripts/convert_annotations.py --splits train valid   # only convert specific splits
    python scripts/convert_annotations.py --max-clips 10         # limit clips (for testing)
    python scripts/convert_annotations.py --sample-rate 5        # use every 5th frame only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


CLASS_MAP: dict[str, int] = {
    "player_left": 0,
    "player_right": 1,
    "goalkeeper_left": 2,
    "goalkeeper_right": 3,
    "referee": 4,
}

YOLO_SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def annotation_to_class_id(role: str, team: str | None) -> int | None:
    """Maps SoccerNet role + team strings to a YOLO class id."""
    role = (role or "").lower().strip()
    team = (team or "").lower().strip()

    if role == "other" or not role:
        return None
    if role == "referee":
        return CLASS_MAP["referee"]

    if role in ("player", "goalkeeper"):
        if team not in ("left", "right"):
            return None
        return CLASS_MAP.get(f"{role}_{team}")

    return None


def _coerce_bbox(raw_bbox) -> list[float] | None:
    """Accepts multiple plausible bbox shapes and returns COCO [x, y, w, h]."""
    if raw_bbox is None:
        return None

    if isinstance(raw_bbox, dict):
        if {"x", "y", "w", "h"}.issubset(raw_bbox):
            return [float(raw_bbox["x"]), float(raw_bbox["y"]), float(raw_bbox["w"]), float(raw_bbox["h"])]
        if {"left", "top", "width", "height"}.issubset(raw_bbox):
            return [
                float(raw_bbox["left"]),
                float(raw_bbox["top"]),
                float(raw_bbox["width"]),
                float(raw_bbox["height"]),
            ]
        return None

    if isinstance(raw_bbox, (list, tuple)):
        if len(raw_bbox) == 4:
            return [float(value) for value in raw_bbox]
        if len(raw_bbox) == 8:
            xs = list(map(float, raw_bbox[0::2]))
            ys = list(map(float, raw_bbox[1::2]))
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            return [x1, y1, x2 - x1, y2 - y1]

    return None


def extract_bbox(annotation: dict) -> list[float] | None:
    """
    Extract a COCO-style [x, y, w, h] bbox from an annotation.

    SoccerNet exports can vary slightly, so we try multiple field names instead
    of assuming only `bbox` exists.
    """
    for key in ("bbox_image", "bbox", "box", "position"):
        if key in annotation:
            bbox = _coerce_bbox(annotation.get(key))
            if bbox is not None:
                return bbox

    attrs = annotation.get("attributes", {})
    for key in ("bbox_image", "bbox", "box", "position"):
        if key in attrs:
            bbox = _coerce_bbox(attrs.get(key))
            if bbox is not None:
                return bbox

    return None


def coco_bbox_to_yolo(bbox_coco: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    """Converts COCO bbox [x_tl, y_tl, w, h] to normalized YOLO format."""
    x_tl, y_tl, w, h = bbox_coco
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None

    cx_norm = (x_tl + w / 2.0) / img_w
    cy_norm = (y_tl + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    if w_norm <= 0 or h_norm <= 0:
        return None

    return (
        max(0.0, min(1.0, cx_norm)),
        max(0.0, min(1.0, cy_norm)),
        max(0.0, min(1.0, w_norm)),
        max(0.0, min(1.0, h_norm)),
    )


def _safe_symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def _resolve_clip_assets(clip_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    """
    Resolve the actual clip root even if the dataset layout has an extra nesting layer.

    Returns (clip_root, label_file, img_dir) or (None, None, None) if not found.
    """
    direct_label = clip_dir / "Labels-GameState.json"
    direct_img_dir = clip_dir / "img1"
    if direct_label.exists() and direct_img_dir.exists():
        return clip_dir, direct_label, direct_img_dir

    for label_file in clip_dir.rglob("Labels-GameState.json"):
        if not label_file.is_file():
            continue
        clip_root = label_file.parent
        img_dir = clip_root / "img1"
        if img_dir.exists():
            return clip_root, label_file, img_dir

        # Some exports keep frames in a sibling/child folder under the clip root.
        for candidate in clip_root.rglob("img1"):
            if candidate.is_dir():
                return clip_root, label_file, candidate

    return None, None, None


def convert_clip(
    clip_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    sample_rate: int = 1,
) -> dict:
    """Converts one SNGS-XXX clip and returns stats."""
    clip_root, label_file, img_dir = _resolve_clip_assets(clip_dir)

    if label_file is None or img_dir is None or clip_root is None:
        log.warning("No clip assets found in %s - skipping clip", clip_dir.name)
        return {}

    with label_file.open() as handle:
        data = json.load(handle)

    version = data.get("info", {}).get("version", "0")
    if str(version) < "1.3":
        log.warning("%s: version %s < 1.3 - annotations may be inaccurate", clip_dir.name, version)

    images_by_id: dict[str, dict] = {}
    for img in data.get("images", []):
        image_id = img.get("image_id") or img.get("id")
        if image_id is None:
            continue
        images_by_id[str(image_id)] = img

    annots_by_image: dict[str, list] = defaultdict(list)
    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id is None:
            continue
        annots_by_image[str(image_id)].append(ann)

    stats = {
        "total_frames": 0,
        "written_frames": 0,
        "total_annots": 0,
        "written_annots": 0,
        "skipped_annots": 0,
    }

    for frame_index, image_id in enumerate(sorted(images_by_id.keys())):
        stats["total_frames"] += 1
        if frame_index % sample_rate != 0:
            continue

        img_meta = images_by_id[image_id]
        img_filename = img_meta.get("file_name")
        img_w = int(img_meta.get("width", 0))
        img_h = int(img_meta.get("height", 0))

        if not img_filename:
            continue

        src_img_path = clip_root / img_filename
        if not src_img_path.exists():
            continue

        frame_stem = f"{clip_root.name}_{Path(img_filename).stem}"
        dst_img_path = out_images_dir / f"{frame_stem}.jpg"
        dst_label_path = out_labels_dir / f"{frame_stem}.txt"

        yolo_lines: list[str] = []
        for ann in annots_by_image.get(image_id, []):
            stats["total_annots"] += 1

            attrs = ann.get("attributes", {})
            class_id = annotation_to_class_id(attrs.get("role", ""), attrs.get("team"))
            if class_id is None:
                stats["skipped_annots"] += 1
                continue

            bbox = extract_bbox(ann)
            if bbox is None:
                stats["skipped_annots"] += 1
                continue

            yolo_bbox = coco_bbox_to_yolo(bbox, img_w, img_h)
            if yolo_bbox is None:
                stats["skipped_annots"] += 1
                continue

            cx, cy, w, h = yolo_bbox
            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats["written_annots"] += 1

        _safe_symlink_or_copy(src_img_path, dst_img_path)
        dst_label_path.write_text(("\n".join(yolo_lines) + "\n") if yolo_lines else "")
        stats["written_frames"] += 1

    return stats


DATASET_YAML_TEMPLATE = """\
# SoccerNet-GSR YOLO Dataset
# Generated by scripts/convert_annotations.py
# Classes: player_left, player_right, goalkeeper_left, goalkeeper_right, referee

path: {dataset_root}
train: images/train
val: images/val
test: images/test

nc: 5
names:
  0: player_left
  1: player_right
  2: goalkeeper_left
  3: goalkeeper_right
  4: referee
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SoccerNet-GSR -> YOLO format")
    parser.add_argument("--data-dir", type=Path, default=Path("data/SoccerNetGS"))
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/soccernet_yolo"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        help="Source splits to convert (valid -> val in YOLO)",
    )
    parser.add_argument("--max-clips", type=int, default=None, help="Max clips per split (None = all)")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=3,
        help="Use every Nth frame (default 3 = ~8fps from 25fps source)",
    )
    args = parser.parse_args()

    total_stats: dict[str, dict] = {}

    for split in args.splits:
        yolo_split = YOLO_SPLIT_MAP.get(split, split)
        split_src = args.data_dir / split

        if not split_src.exists():
            log.error("Split directory not found: %s", split_src)
            continue

        out_images = args.out_dir / "images" / yolo_split
        out_labels = args.out_dir / "labels" / yolo_split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        clip_dirs = sorted([path for path in split_src.iterdir() if path.is_dir()])
        if args.max_clips is not None:
            clip_dirs = clip_dirs[: args.max_clips]

        log.info("Converting split '%s' (%d clips) -> '%s'", split, len(clip_dirs), yolo_split)
        split_stats = {
            "total_frames": 0,
            "written_frames": 0,
            "total_annots": 0,
            "written_annots": 0,
            "skipped_annots": 0,
        }

        for clip_dir in tqdm(clip_dirs, desc=split, unit="clip"):
            stats = convert_clip(clip_dir, out_images, out_labels, sample_rate=max(1, args.sample_rate))
            for key in split_stats:
                split_stats[key] += stats.get(key, 0)

        total_stats[split] = split_stats
        log.info(
            "  %s: %d frames written, %d annotations (%d skipped)",
            split,
            split_stats["written_frames"],
            split_stats["written_annots"],
            split_stats["skipped_annots"],
        )

    yaml_path = args.out_dir / "dataset.yaml"
    yaml_path.write_text(DATASET_YAML_TEMPLATE.format(dataset_root=str(args.out_dir.resolve())))
    log.info("Wrote dataset config: %s", yaml_path)

    log.info("\n=== Conversion Summary ===")
    for split, stats in total_stats.items():
        log.info(
            "  %-10s: %6d frames | %7d annotations written | %6d skipped",
            split,
            stats["written_frames"],
            stats["written_annots"],
            stats["skipped_annots"],
        )
    log.info("  Output: %s", args.out_dir.resolve())


if __name__ == "__main__":
    main()