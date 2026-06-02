"""Download a YOLO model weight file from a remote URL.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --url <url> --output models/my_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt"
)
DEFAULT_OUTPUT = "yolov8m_fixed.pt"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"         -> {dest}")

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with dest.open("wb") as fh:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    done = int(50 * downloaded / total)
                    bar = "=" * done + " " * (50 - done)
                    mb_done = downloaded / 1_048_576
                    mb_total = total / 1_048_576
                    sys.stdout.write(f"\r[{bar}] {mb_done:.1f} / {mb_total:.1f} MB")
                    sys.stdout.flush()

    print(f"\nDownload complete: {dest}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a YOLO model weight file")
    parser.add_argument("--url",    default=DEFAULT_URL,    help="Remote URL of the model file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Local destination path")
    args = parser.parse_args(argv)

    try:
        download_file(url=args.url, dest=Path(args.output))
    except requests.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"File error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
