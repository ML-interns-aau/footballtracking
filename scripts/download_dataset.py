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

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data/SoccerNetGS")
TASK_NAME = "gamestate-2024"
REQUIRED_VERSION = "1.3"


def download_splits(splits: list[str], local_dir: Path) -> None:
    """Downloads the requested dataset splits using the SoccerNet downloader."""
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError as exc:
        raise ImportError("Install SoccerNet: pip install SoccerNet") from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    downloader = SoccerNetDownloader(LocalDirectory=str(local_dir))
    log.info("Downloading splits: %s", ", ".join(splits))
    downloader.downloadDataTask(task=TASK_NAME, split=splits)
    log.info("Download complete. Unzipping...")
    _unzip_splits(splits, local_dir)


def _unzip_splits(splits: list[str], local_dir: Path) -> None:
    """Unzips downloaded zip files into their respective split directories."""
    for split in splits:
        zip_path = local_dir / TASK_NAME / f"{split}.zip"
        out_dir = local_dir / split
        if not zip_path.exists():
            log.warning("Zip not found: %s - skipping", zip_path)
            continue
        if out_dir.exists() and any(out_dir.iterdir()):
            log.info("%s/ already unzipped - skipping", split)
            continue
        log.info("Unzipping %s -> %s", zip_path, out_dir)
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
            log.error("Missing split directory: %s", split_dir)
            all_ok = False
            continue

        clip_dirs = sorted([path for path in split_dir.iterdir() if path.is_dir()])
        if not clip_dirs:
            log.error("Split directory is empty: %s", split_dir)
            all_ok = False
            continue

        log.info("%s: %d clips found", split, len(clip_dirs))

        # Spot-check first clip
        first_clip = clip_dirs[0]
        label_file = first_clip / "Labels-GameState.json"
        img_dir = first_clip / "img1"

        if not label_file.exists():
            log.error("Missing labels: %s", label_file)
            all_ok = False
            continue

        with label_file.open() as f:
            data = json.load(f)

        version = data.get("info", {}).get("version", "unknown")
        if version < REQUIRED_VERSION:
            log.error(
                "%s: Labels version %s < required %s. Re-download the dataset.",
                first_clip.name,
                version,
                REQUIRED_VERSION,
            )
            all_ok = False

        n_images = len(data.get("images", []))
        n_annotations = len(data.get("annotations", []))
        n_frames_disk = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0

        log.info(
            "  %s: v%s | %d image records | %d annotations | %d frames on disk",
            first_clip.name,
            version,
            n_images,
            n_annotations,
            n_frames_disk,
        )

        # Verify annotation fields exist
        annotations = data.get("annotations", [])
        if annotations:
            sample = annotations[0]
            for required_key in ("bbox", "attributes", "track_id", "image_id"):
                if required_key not in sample:
                    log.error("Annotation missing field '%s' in %s", required_key, first_clip.name)
                    all_ok = False

            attrs = sample.get("attributes", {})
            if "role" not in attrs:
                log.error("Annotation 'attributes' missing 'role' in %s", first_clip.name)
                all_ok = False

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify SoccerNet-GSR dataset")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test", "challenge"],
        help="Dataset splits to download",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Local directory to store dataset",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip download, only verify existing data",
    )
    args = parser.parse_args()

    if not args.verify_only:
        download_splits(args.splits, args.data_dir)

    log.info("Verifying dataset structure...")
    ok = verify_dataset(args.data_dir, args.splits)
    if ok:
        log.info("\u2713 Dataset verified successfully.")
    else:
        log.error("\u2717 Dataset verification failed. Check errors above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()