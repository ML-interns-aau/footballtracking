"""
Resilient dependency installer.
Installs each package individually with retries so a dropped connection
on one package doesn't block the rest.
Run with: venv\Scripts\python.exe install_deps.py
"""
import subprocess
import sys
import time

PIP = [sys.executable, "-m", "pip", "install", "--retries", "10", "--timeout", "120"]

PACKAGES = [
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

def install(pkg, attempt=1, max_attempts=5):
    print(f"\n{'='*50}")
    print(f"Installing: {pkg}  (attempt {attempt}/{max_attempts})")
    print('='*50)
    result = subprocess.run(PIP + [pkg])
    if result.returncode == 0:
        print(f"  ✓ {pkg} installed successfully")
        return True
    if attempt < max_attempts:
        wait = attempt * 5
        print(f"  ✗ Failed. Retrying in {wait}s...")
        time.sleep(wait)
        return install(pkg, attempt + 1, max_attempts)
    print(f"  ✗ {pkg} failed after {max_attempts} attempts — skipping")
    return False

failed = []
for pkg in PACKAGES:
    ok = install(pkg)
    if not ok:
        failed.append(pkg)

print("\n" + "="*50)
print("INSTALL COMPLETE")
print("="*50)
if failed:
    print(f"Failed packages: {failed}")
    print("Re-run this script to retry them.")
else:
    print("All packages installed successfully!")

print("\nVerifying key imports...")
checks = [
    ("torch",       "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"),
    ("streamlit",   "import streamlit; print('streamlit', streamlit.__version__)"),
    ("cv2",         "import cv2; print('opencv', cv2.__version__)"),
    ("ultralytics", "import ultralytics; print('ultralytics', ultralytics.__version__)"),
    ("numpy",       "import numpy; print('numpy', numpy.__version__)"),
    ("pandas",      "import pandas; print('pandas', pandas.__version__)"),
    ("supervision", "import supervision; print('supervision', supervision.__version__)"),
]
for name, code in checks:
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  ✓ {r.stdout.strip()}")
    else:
        print(f"  ✗ {name}: {r.stderr.strip()[:80]}")

print("\nDone! Run the app with:")
print("  venv\\Scripts\\streamlit.exe run dashboard/Home.py")
