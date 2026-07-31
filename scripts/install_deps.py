"""Install all project dependencies with automatic retry logic.

Usage:
    python scripts/install_deps.py
    python scripts/install_deps.py --skip-verify
"""

from __future__ import annotations

import subprocess
import sys
import time
import argparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PIP = [sys.executable, "-m", "pip", "install", "--retries", "10", "--timeout", "120"]

PACKAGES: list[str] = [
    "typing_extensions",
    "numpy",
    "pandas",
    "matplotlib",
    "plotly",
    "requests",
    "tqdm",
    "psutil",
    "seaborn",
    "scikit-learn",
    "pillow",
    "opencv-python-headless",
    "scipy",
    "supervision",
    "lapx",
    "ultralytics",
    "streamlit",
]

_VERIFY_CHECKS: list[tuple[str, str]] = [
    ("torch",       "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"),
    ("streamlit",   "import streamlit; print('streamlit', streamlit.__version__)"),
    ("cv2",         "import cv2; print('opencv', cv2.__version__)"),
    ("ultralytics", "import ultralytics; print('ultralytics', ultralytics.__version__)"),
    ("numpy",       "import numpy; print('numpy', numpy.__version__)"),
    ("pandas",      "import pandas; print('pandas', pandas.__version__)"),
    ("supervision", "import supervision; print('supervision', supervision.__version__)"),
]

_SEP = "=" * 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install(pkg: str, attempt: int = 1, max_attempts: int = 5) -> bool:
    print(f"\n{_SEP}")
    print(f"Installing: {pkg}  (attempt {attempt}/{max_attempts})")
    print(_SEP)
    result = subprocess.run(_PIP + [pkg])
    if result.returncode == 0:
        print(f"  ✓ {pkg} installed successfully")
        return True
    if attempt < max_attempts:
        wait = attempt * 5
        print(f"  ✗ Failed. Retrying in {wait}s...")
        time.sleep(wait)
        return _install(pkg, attempt + 1, max_attempts)
    print(f"  ✗ {pkg} failed after {max_attempts} attempts — skipping")
    return False


def _verify_imports() -> None:
    print("\nVerifying key imports...")
    for name, code in _VERIFY_CHECKS:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  ✓ {r.stdout.strip()}")
        else:
            print(f"  ✗ {name}: {r.stderr.strip()[:80]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install project dependencies")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the post-install import verification step",
    )
    args = parser.parse_args(argv)

    failed: list[str] = []
    for pkg in PACKAGES:
        if not _install(pkg):
            failed.append(pkg)

    print(f"\n{_SEP}")
    print("INSTALL COMPLETE")
    print(_SEP)

    if failed:
        print(f"Failed packages: {failed}")
        print("Re-run this script to retry them.")
    else:
        print("All packages installed successfully!")

    if not args.skip_verify:
        _verify_imports()

    print("\nDone! Run the app with:")
    print("  venv\\Scripts\\streamlit.exe run app/Home.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
