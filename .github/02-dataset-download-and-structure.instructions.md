# Phase 2 — SoccerNet-GSR Dataset Download and Structure
# File: .github/instructions/02-dataset-download-and-structure.instructions.md
# Apply to: scripts/download_dataset.py

## Goal

Download the SoccerNet Game State Reconstruction (GSR) dataset v1.3 and verify
its structure. This dataset is the source of all annotations used in phase 3 and 4.

---

## What the Dataset Contains

SoccerNet-GSR contains 200 thirty-second video clips from broadcast football matches,
split into:
- `train/`: 57 clips (SNGS-001 through SNGS-057 range)
- `valid/`: 59 clips
- `test/`:  50 clips (labels available)
- `challenge/`: 34 clips (labels withheld for benchmark)

Each clip folder contains:
```
SNGS-XXX/
├── Labels-GameState.json     ← all annotations for this clip
├── img1/                     ← extracted frames as JPEG images
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...  (up to ~750 frames at 25fps for 30s clips)
└── gamestate-2024/           ← metadata (seq info, etc.)
```

`Labels-GameState.json` follows a COCO-like structure. Critical fields per annotation:
```json
{
  "annotations": [
    {
      "id": 12345,
      "image_id": 1,
      "bbox": [x, y, width, height],      ← COCO format: top-left x,y + w,h in pixels
      "attributes": {
        "role": "player",                  ← "player" | "goalkeeper" | "referee" | "other"
        "team": "left",                    ← "left" | "right" (only for player/goalkeeper)
        "jersey": "7"                      ← jersey number string, may be null
      },
      "track_id": 3                        ← persistent ID within the clip
    }
  ],
  "images": [
    {
      "id": 1,
      "file_name": "img1/000001.jpg",
      "width": 1920,
      "height": 1080,
      "frame_id": 1
    }
  ],
  "info": {
    "version": "1.3"                       ← must be >= 1.3
  }
}
```

---

## Download Script

Create `scripts/download_dataset.py`:

```python
"""
Downloads the SoccerNet-GSR dataset v1.3 to data/SoccerNetGS/.

Requirements:
    pip install SoccerNet tqdm

Usage:
    python scripts/download_dataset.py --splits train valid test
    python scripts/download_dataset.py --splits train valid test challenge
    python scripts/download_dataset.py --verify-only   # just check existing download

The SoccerNet package will prompt for your credentials on first use.
Register for free at: https://www.soccer-net.org/data
"""
import argparse
import json
import zipfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data/SoccerNetGS")
TASK_NAME = "gamestate-2024"
REQUIRED_VERSION = "1.3"


def download_splits(splits: list[str], local_dir: Path) -> None:
    """Downloads the requested dataset splits using the SoccerNet downloader."""
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        raise ImportError("Install SoccerNet: pip install SoccerNet")

    local_dir.mkdir(parents=True, exist_ok=True)
    downloader = SoccerNetDownloader(LocalDirectory=str(local_dir))
    log.info(f"Downloading splits: {splits}")
    downloader.downloadDataTask(task=TASK_NAME, split=splits)
    log.info("Download complete. Unzipping...")
    _unzip_splits(splits, local_dir)


def _unzip_splits(splits: list[str], local_dir: Path) -> None:
    """Unzips downloaded zip files into their respective split directories."""
    for split in splits:
        zip_path = local_dir / TASK_NAME / f"{split}.zip"
        out_dir  = local_dir / split
        if not zip_path.exists():
            log.warning(f"Zip not found: {zip_path} — skipping")
            continue
        if out_dir.exists() and any(out_dir.iterdir()):
            log.info(f"{split}/ already unzipped — skipping")
            continue
        log.info(f"Unzipping {zip_path} → {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    log.info("Unzip complete.")


def verify_dataset(local_dir: Path, splits: list[str]) -> bool:
    """
    Verifies dataset structure and annotation version for each split.
    Returns True if everything looks correct, False otherwise.
    """
    all_ok = True
    for split in splits:
        split_dir = local_dir / split
        if not split_dir.exists():
            log.error(f"Missing split directory: {split_dir}")
            all_ok = False
            continue

        clip_dirs = sorted(split_dir.iterdir())
        if not clip_dirs:
            log.error(f"Split directory is empty: {split_dir}")
            all_ok = False
            continue

        log.info(f"{split}: {len(clip_dirs)} clips found")

        # Spot-check first clip
        first_clip = clip_dirs[0]
        label_file = first_clip / "Labels-GameState.json"
        img_dir    = first_clip / "img1"

        if not label_file.exists():
            log.error(f"Missing labels: {label_file}")
            all_ok = False
            continue

        with open(label_file) as f:
            data = json.load(f)

        version = data.get("info", {}).get("version", "unknown")
        if version < REQUIRED_VERSION:
            log.error(
                f"{first_clip.name}: Labels version {version} < required {REQUIRED_VERSION}. "
                f"Re-download the dataset."
            )
            all_ok = False

        n_images      = len(data.get("images", []))
        n_annotations = len(data.get("annotations", []))
        n_frames_disk = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0

        log.info(
            f"  {first_clip.name}: v{version} | "
            f"{n_images} image records | {n_annotations} annotations | "
            f"{n_frames_disk} frames on disk"
        )

        # Verify annotation fields exist
        if data["annotations"]:
            sample = data["annotations"][0]
            for required_key in ("bbox", "attributes", "track_id", "image_id"):
                if required_key not in sample:
                    log.error(f"Annotation missing field '{required_key}' in {first_clip.name}")
                    all_ok = False

            attrs = sample.get("attributes", {})
            if "role" not in attrs:
                log.error(f"Annotation 'attributes' missing 'role' in {first_clip.name}")
                all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Download and verify SoccerNet-GSR dataset")
    parser.add_argument(
        "--splits", nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test", "challenge"],
        help="Dataset splits to download"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Local directory to store dataset"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Skip download, only verify existing data"
    )
    args = parser.parse_args()

    if not args.verify_only:
        download_splits(args.splits, args.data_dir)

    log.info("Verifying dataset structure...")
    ok = verify_dataset(args.data_dir, args.splits)
    if ok:
        log.info("✓ Dataset verified successfully.")
    else:
        log.error("✗ Dataset verification failed. Check errors above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

---

## Registration Required

SoccerNet requires free registration before download. Direct the user to:
`https://www.soccer-net.org/data`

When `SoccerNetDownloader` runs for the first time, it will prompt for
NDA acceptance and credentials. This is a one-time step.

---

## Expected Dataset Size

| Split | Clips | Approx size on disk |
|---|---|---|
| train | 57 | ~18 GB |
| valid | 59 | ~19 GB |
| test  | 50 | ~16 GB |
| challenge | 34 | ~11 GB |
| **Total** | **200** | **~64 GB** |

If disk space is limited, download only `train` and `valid` for fine-tuning.
The `test` split is used for final evaluation only.

---

## Verification Checklist

After running the download script:
- [ ] `data/SoccerNetGS/train/` contains at least 50 subdirectories named `SNGS-XXX`
- [ ] Each `SNGS-XXX/` has `Labels-GameState.json` and an `img1/` folder
- [ ] `Labels-GameState.json` has `info.version >= 1.3`
- [ ] `img1/` contains `.jpg` frame files (typically 600–750 per clip)
- [ ] Script exits with `✓ Dataset verified successfully.`
