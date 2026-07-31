"""Tests for src/engine/camera_motion.py (CameraMotionEstimator).

Self-contained: exercises the estimator on a synthetic pure-pan sequence, and
(if clip_1.mp4 is present at the repo root) asserts the pan-shape signature
found by manual analysis — near-zero pan before the run-up, a clear pan peak
during frames 94-118 (the camera following the corner-taker's approach).
Run directly (``python test_camera_motion.py``) or via pytest.
"""
from pathlib import Path

import cv2
import numpy as np

from src.engine.camera_motion import CameraMotionEstimator

REPO_ROOT = Path(__file__).resolve().parent
CLIP_1 = REPO_ROOT / "clip_1.mp4"


def _synthetic_pan_frames(shift_px: tuple[float, float], size: int = 240, n: int = 5):
    """A textured frame plus n copies translated by shift_px, for a controlled pan test."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(size + 40, size + 40), dtype=np.uint8)
    frames = []
    for i in range(n):
        ox, oy = int(round(shift_px[0] * i)), int(round(shift_px[1] * i))
        frames.append(base[20 - oy:20 - oy + size, 20 - ox:20 - ox + size].copy())
    return frames


def test_synthetic_pan_recovered():
    shift = (2.0, -1.0)
    frames = _synthetic_pan_frames(shift)
    est = CameraMotionEstimator()
    dx, dy, M = est.estimate(frames[0], frames[1])
    assert M is not None, "expected a fitted motion matrix on a clean synthetic pan"
    assert abs(dx - shift[0]) < 0.6, f"dx={dx} expected ~{shift[0]}"
    assert abs(dy - shift[1]) < 0.6, f"dy={dy} expected ~{shift[1]}"
    print(f"[OK] synthetic pan recovered: dx={dx:.2f} dy={dy:.2f}")


def test_degrades_gracefully_on_blank_frames():
    blank_prev = np.zeros((100, 100), dtype=np.uint8)
    blank_curr = np.zeros((100, 100), dtype=np.uint8)
    est = CameraMotionEstimator()
    dx, dy, M = est.estimate(blank_prev, blank_curr)
    assert (dx, dy, M) == (0.0, 0.0, None)
    print("[OK] degrades gracefully with <10 trackable points")


def test_player_mask_excludes_region():
    shift = (3.0, 0.0)
    frames = _synthetic_pan_frames(shift, size=200)
    est = CameraMotionEstimator()
    h, w = frames[0].shape[:2]
    # Mask out the whole frame except a 1px sliver -> should degrade to no-motion.
    dx, dy, M = est.estimate(frames[0], frames[1], player_boxes=[(0, 0, w, h - 1)])
    assert (dx, dy, M) == (0.0, 0.0, None)
    print("[OK] player_boxes masking removes feature candidates as expected")


def test_clip1_pan_shape():
    if not CLIP_1.exists():
        print("[SKIP] clip_1.mp4 not found at repo root; skipping real-clip pan-shape test")
        return

    cap = cv2.VideoCapture(str(CLIP_1))
    est = CameraMotionEstimator()
    ok, frame = cap.read()
    assert ok, "could not read first frame of clip_1.mp4"
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    magnitudes = [0.0]  # index 0 = frame 0, no predecessor
    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dx, dy, _ = est.estimate(prev_gray, curr_gray)
        magnitudes.append(float(np.hypot(dx, dy)))
        prev_gray = curr_gray
        idx += 1
    cap.release()

    magnitudes = np.array(magnitudes)
    assert len(magnitudes) > 118, f"expected >118 frames, got {len(magnitudes)}"

    pre_run_up = magnitudes[1:91]
    pan_window = magnitudes[94:119]
    assert pre_run_up.max() < 1.0, f"expected <1.0 px/f before frame 91, got max={pre_run_up.max():.2f}"
    assert pan_window.max() > 2.0, f"expected >2.0 px/f somewhere in 94-118, got max={pan_window.max():.2f}"
    print(
        f"[OK] clip_1 pan shape: pre-run-up max={pre_run_up.max():.2f} px/f, "
        f"94-118 max={pan_window.max():.2f} px/f"
    )


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{'='*50}\nALL {len(tests)} CAMERA-MOTION TESTS PASSED!\n{'='*50}")


if __name__ == "__main__":
    run_all()
