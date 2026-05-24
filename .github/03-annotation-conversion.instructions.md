# Phase 3 — Annotation Conversion: SoccerNet-GSR JSON → YOLO Format
# File: .github/instructions/03-annotation-conversion.instructions.md
# Apply to: scripts/convert_annotations.py

## Goal

Convert `Labels-GameState.json` annotations from all SoccerNet-GSR clips into
the YOLO label format required to fine-tune an Ultralytics YOLO model.

The output is a self-contained dataset directory at `datasets/soccernet_yolo/`
that YOLO can consume directly via a `dataset.yaml` config file.

---

## Class Mapping (from copilot-instructions.md)

```python
CLASS_MAP = {
    "player_left":       0,
    "player_right":      1,
    "goalkeeper_left":   2,
    "goalkeeper_right":  3,
    "referee":           4,
}
# role=other → SKIP entirely (coaches, staff, etc.)
```

SoccerNet annotation → YOLO class:
- `role=player,     team=left`  → class 0
- `role=player,     team=right` → class 1
- `role=goalkeeper, team=left`  → class 2
- `role=goalkeeper, team=right` → class 3
- `role=referee,    team=any`   → class 4
- `role=other`                  → **skip** (not written to label file)

---

## YOLO Label Format

Each image gets one `.txt` file. Each line in the file is one bounding box:
```
<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
```
- All values are floats in `[0.0, 1.0]` normalized by image width/height
- `cx`, `cy` are the **center** of the bbox (not top-left corner)
- SoccerNet uses COCO bbox format `[x, y, w, h]` where `x,y` is the **top-left** corner

Conversion from SoccerNet COCO to YOLO:
```python
# SoccerNet: x_tl, y_tl, w, h  (absolute pixels)
# YOLO:      cx, cy, w, h      (normalized 0-1)
cx_norm = (x_tl + w / 2) / img_width
cy_norm = (y_tl + h / 2) / img_height
w_norm  = w / img_width
h_norm  = h / img_height
```

---

## Conversion Script

Create `scripts/convert_annotations.py`:

```python
"""
Converts SoccerNet-GSR Labels-GameState.json annotations to YOLO format.

For each clip, reads the JSON, maps role+team → class_id, converts COCO bboxes
to YOLO normalized format, and writes one .txt label file per image frame.
Images are symlinked (not copied) to save disk space.

Usage:
    python scripts/convert_annotations.py
    python scripts/convert_annotations.py --data-dir data/SoccerNetGS --out-dir datasets/soccernet_yolo
    python scripts/convert_annotations.py --splits train valid   # only convert specific splits
    python scripts/convert_annotations.py --max-clips 10         # limit clips (for testing)
    python scripts/convert_annotations.py --sample-rate 5        # use every 5th frame only
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Class mapping ─────────────────────────────────────────────────────────────

CLASS_MAP: dict[str, int] = {
    "player_left":       0,
    "player_right":      1,
    "goalkeeper_left":   2,
    "goalkeeper_right":  3,
    "referee":           4,
}

def annotation_to_class_id(role: str, team: str | None) -> int | None:
    """
    Maps a SoccerNet role + team string to a YOLO class id.
    Returns None for annotations that should be skipped (role=other, missing team).
    """
    role = role.lower().strip()
    team = (team or "").lower().strip()

    if role == "other":
        return None  # coaches, staff — skip

    if role == "referee":
        return CLASS_MAP["referee"]

    if role in ("player", "goalkeeper"):
        if team not in ("left", "right"):
            # Annotation has missing/invalid team — skip
            return None
        key = f"{role}_{team}"
        return CLASS_MAP.get(key)

    return None  # unknown role — skip


def coco_bbox_to_yolo(bbox_coco: list, img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    """
    Converts COCO bbox [x_tl, y_tl, w, h] (absolute pixels) to
    YOLO format (cx_norm, cy_norm, w_norm, h_norm) normalized to [0,1].

    Returns None if the bbox is degenerate (zero area or out of bounds).
    """
    x_tl, y_tl, w, h = bbox_coco
    if w <= 0 or h <= 0:
        return None

    cx = x_tl + w / 2.0
    cy = y_tl + h / 2.0

    cx_norm = cx / img_w
    cy_norm = cy / img_h
    w_norm  = w  / img_w
    h_norm  = h  / img_h

    # Clamp to valid range (handles bbox slightly outside image boundaries)
    cx_norm = max(0.0, min(1.0, cx_norm))
    cy_norm = max(0.0, min(1.0, cy_norm))
    w_norm  = max(0.0, min(1.0, w_norm))
    h_norm  = max(0.0, min(1.0, h_norm))

    if w_norm < 1e-4 or h_norm < 1e-4:
        return None  # effectively zero area after clamping

    return cx_norm, cy_norm, w_norm, h_norm


# ── Per-clip conversion ───────────────────────────────────────────────────────

def convert_clip(
    clip_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    sample_rate: int = 1,
) -> dict:
    """
    Converts one SNGS-XXX clip.

    Returns stats dict: {total_frames, written_frames, total_annots, written_annots, skipped_annots}
    """
    label_file = clip_dir / "Labels-GameState.json"
    img_dir    = clip_dir / "img1"

    if not label_file.exists():
        log.warning(f"No Labels-GameState.json in {clip_dir.name} — skipping clip")
        return {}
    if not img_dir.exists():
        log.warning(f"No img1/ directory in {clip_dir.name} — skipping clip")
        return {}

    with open(label_file) as f:
        data = json.load(f)

    # Check annotation version
    version = data.get("info", {}).get("version", "0")
    if version < "1.3":
        log.warning(f"{clip_dir.name}: version {version} < 1.3 — annotations may be inaccurate")

    # Build image_id → image metadata lookup
    images_by_id: dict[int, dict] = {img["id"]: img for img in data.get("images", [])}

    # Group annotations by image_id for efficient frame-level processing
    annots_by_image: dict[int, list] = defaultdict(list)
    for ann in data.get("annotations", []):
        annots_by_image[ann["image_id"]].append(ann)

    stats = {"total_frames": 0, "written_frames": 0, "total_annots": 0, "written_annots": 0, "skipped_annots": 0}
    sorted_image_ids = sorted(images_by_id.keys())

    for i, image_id in enumerate(sorted_image_ids):
        stats["total_frames"] += 1

        # Apply frame sampling
        if i % sample_rate != 0:
            continue

        img_meta = images_by_id[image_id]
        img_filename = img_meta["file_name"]          # e.g. "img1/000001.jpg"
        img_w = img_meta["width"]
        img_h = img_meta["height"]

        # Source image path on disk
        src_img_path = clip_dir / img_filename
        if not src_img_path.exists():
            continue

        # Build output paths — use clip name + frame name to avoid collisions
        frame_stem = f"{clip_dir.name}_{Path(img_filename).stem}"
        dst_img_path   = out_images_dir / f"{frame_stem}.jpg"
        dst_label_path = out_labels_dir / f"{frame_stem}.txt"

        # Convert annotations for this frame
        yolo_lines: list[str] = []
        for ann in annots_by_image.get(image_id, []):
            stats["total_annots"] += 1

            attrs = ann.get("attributes", {})
            role  = attrs.get("role", "")
            team  = attrs.get("team", None)

            class_id = annotation_to_class_id(role, team)
            if class_id is None:
                stats["skipped_annots"] += 1
                continue

            yolo_bbox = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            if yolo_bbox is None:
                stats["skipped_annots"] += 1
                continue

            cx, cy, w, h = yolo_bbox
            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats["written_annots"] += 1

        # Only write frames that have at least one valid annotation
        if not yolo_lines:
            stats["skipped_annots"] += len(annots_by_image.get(image_id, []))
            continue

        # Symlink image (saves ~64 GB of disk space vs copying)
        if not dst_img_path.exists():
            try:
                os.symlink(src_img_path.resolve(), dst_img_path)
            except OSError:
                shutil.copy2(src_img_path, dst_img_path)  # fallback: copy

        # Write label file
        dst_label_path.write_text("\n".join(yolo_lines) + "\n")
        stats["written_frames"] += 1

    return stats


# ── Dataset YAML ─────────────────────────────────────────────────────────────

DATASET_YAML_TEMPLATE = """\
# SoccerNet-GSR YOLO Dataset
# Generated by scripts/convert_annotations.py
# Classes: player_left, player_right, goalkeeper_left, goalkeeper_right, referee

path: {dataset_root}
train: images/train
val:   images/val

nc: 5
names:
  0: player_left
  1: player_right
  2: goalkeeper_left
  3: goalkeeper_right
  4: referee
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert SoccerNet-GSR → YOLO format")
    parser.add_argument("--data-dir",    type=Path, default=Path("data/SoccerNetGS"))
    parser.add_argument("--out-dir",     type=Path, default=Path("datasets/soccernet_yolo"))
    parser.add_argument("--splits",      nargs="+", default=["train", "valid"],
                        help="Source splits to convert (valid → val in YOLO)")
    parser.add_argument("--max-clips",   type=int,  default=None,
                        help="Max clips per split (None = all). Use for smoke testing.")
    parser.add_argument("--sample-rate", type=int,  default=3,
                        help="Use every Nth frame (default 3 = ~8fps from 25fps source). "
                             "Use 1 for all frames (large dataset), 5 for quick experiments.")
    args = parser.parse_args()

    # Split name mapping: SoccerNet "valid" → YOLO "val"
    YOLO_SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}

    total_stats: dict[str, dict] = {}

    for split in args.splits:
        yolo_split = YOLO_SPLIT_MAP.get(split, split)
        split_src  = args.data_dir / split

        if not split_src.exists():
            log.error(f"Split directory not found: {split_src}")
            continue

        out_images = args.out_dir / "images" / yolo_split
        out_labels = args.out_dir / "labels" / yolo_split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        clip_dirs = sorted(split_src.iterdir())
        if args.max_clips:
            clip_dirs = clip_dirs[: args.max_clips]

        log.info(f"Converting split '{split}' ({len(clip_dirs)} clips) → '{yolo_split}'")
        split_stats = {"total_frames": 0, "written_frames": 0, "total_annots": 0,
                       "written_annots": 0, "skipped_annots": 0}

        for clip_dir in tqdm(clip_dirs, desc=f"{split}", unit="clip"):
            if not clip_dir.is_dir():
                continue
            stats = convert_clip(clip_dir, out_images, out_labels, sample_rate=args.sample_rate)
            for k in split_stats:
                split_stats[k] += stats.get(k, 0)

        total_stats[split] = split_stats
        log.info(
            f"  {split}: {split_stats['written_frames']} frames written, "
            f"{split_stats['written_annots']} annotations "
            f"({split_stats['skipped_annots']} skipped)"
        )

    # Write dataset.yaml
    yaml_path = args.out_dir / "dataset.yaml"
    yaml_path.write_text(DATASET_YAML_TEMPLATE.format(
        dataset_root=str(args.out_dir.resolve())
    ))
    log.info(f"Wrote dataset config: {yaml_path}")

    # Summary
    log.info("\n=== Conversion Summary ===")
    for split, s in total_stats.items():
        log.info(
            f"  {split:10s}: {s['written_frames']:6d} frames | "
            f"{s['written_annots']:7d} annotations written | "
            f"{s['skipped_annots']:6d} skipped"
        )
    log.info(f"  Output: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
```

---

## Verifying the Conversion

After running the script, run this quick verification to check label distribution:

```python
# Run as: python scripts/verify_conversion.py
from pathlib import Path
from collections import Counter

label_dir = Path("datasets/soccernet_yolo/labels/train")
class_names = ["player_left", "player_right", "goalkeeper_left", "goalkeeper_right", "referee"]

counts = Counter()
n_files = 0
for lf in label_dir.glob("*.txt"):
    n_files += 1
    for line in lf.read_text().strip().splitlines():
        cls = int(line.split()[0])
        counts[cls] += 1

print(f"Label files: {n_files}")
for i, name in enumerate(class_names):
    print(f"  class {i} ({name}): {counts[i]:,}")
```

Expected output (approximate, varies by sample rate):
```
Label files: ~12,000–40,000
  class 0 (player_left):       ~150,000
  class 1 (player_right):      ~150,000
  class 2 (goalkeeper_left):     ~8,000
  class 3 (goalkeeper_right):    ~8,000
  class 4 (referee):            ~25,000
```

Note: classes 2 and 3 (goalkeepers) will be significantly underrepresented compared
to outfield players. This is addressed in phase 4 via class weights.

---

## Recommended Sample Rate

| `--sample-rate` | Approx frames (train) | Dataset size | Use case |
|---|---|---|---|
| 1 | ~1.1M | ~80 GB | Maximum data (slow) |
| 3 | ~370K | ~27 GB | **Recommended** |
| 5 | ~220K | ~16 GB | Fast experiments |
| 10 | ~110K | ~8 GB | Smoke test only |

Start with `--sample-rate 3 --max-clips 10` to verify the pipeline before running
the full conversion.
