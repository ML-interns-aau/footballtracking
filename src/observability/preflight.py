from __future__ import annotations
import shutil
from pathlib import Path
from typing import Any
from .errors import (
    ModelNotFoundError,
    VideoOpenError,
    VideoEmptyError,
    VideoCorruptError,
    ConfigurationError,
    ExportError,
)
_MIN_FREE_BYTES = 500 * 1024 * 1024
def run_preflight(
    *,
    model_path:  str,
    input_path:  str,
    output_dir:  str,
    min_free_mb: int = 500,
) -> dict[str, Any]:
    _check_model(model_path)
    _check_output_dir(output_dir, min_free_mb)
    meta = _check_video(input_path)
    return meta
def _check_model(model_path: str) -> None:
    p = Path(model_path)
    if not p.exists():
        raise ModelNotFoundError(str(p))
    if p.stat().st_size == 0:
        raise ModelNotFoundError(f"{p} (file is empty — download may have been interrupted)")
def _check_output_dir(output_dir: str, min_free_mb: int) -> None:
    out = Path(output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ExportError(str(out), reason=f"Cannot create output directory: {e}") from e
    probe = out / ".write_probe"
    try:
        probe.touch()
        probe.unlink()
    except OSError as e:
        raise ExportError(str(out), reason=f"Output directory is not writable: {e}") from e
    usage = shutil.disk_usage(out)
    min_bytes = min_free_mb * 1024 * 1024
    if usage.free < min_bytes:
        free_mb = usage.free // (1024 * 1024)
        raise ExportError(
            str(out),
            reason=(
                f"Insufficient disk space: {free_mb} MB free, "
                f"{min_free_mb} MB required."
            ),
        )
def _check_video(input_path: str) -> dict[str, Any]:
    import cv2
    p = Path(input_path)
    if not p.exists():
        raise VideoOpenError(str(p))
    if p.stat().st_size == 0:
        raise VideoOpenError(f"{p} (file is empty)")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap.release()
        raise VideoOpenError(str(p))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        ret, _ = cap.read()
        cap.release()
        if not ret:
            raise VideoEmptyError(str(p))
        return {"width": width, "height": height, "fps": fps, "total_frames": -1}
    ret, _ = cap.read()
    cap.release()
    if not ret:
        raise VideoCorruptError(str(p), frame_idx=0)
    return {
        "width":        width,
        "height":       height,
        "fps":          fps,
        "total_frames": total_frames,
    }