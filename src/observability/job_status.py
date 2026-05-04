"""
Live job status tracker — writes a job_status.json file to the output
directory that is updated throughout the pipeline run.

The UI (or any external monitor) can poll this file to show real-time
progress without needing a long-lived process connection.

Status values
-------------
  pending   → run created but not yet started
  running   → pipeline is actively processing frames
  success   → completed with all planned frames processed
  partial   → completed but fewer frames than planned were processed
  failure   → run aborted with an unrecoverable error
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JobStatus:
    """
    Writes an up-to-date job_status.json on every update() call.

    Designed for single-threaded use inside the main pipeline loop.
    Writes are best-effort — I/O errors are silently ignored so they
    never crash the pipeline itself.
    """

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_SUCCESS = "success"
    STATUS_PARTIAL = "partial"
    STATUS_FAILURE = "failure"

    def __init__(self, output_dir: str, run_id: str) -> None:
        self._path      = Path(output_dir) / "job_status.json"
        self._run_id    = run_id
        self._started   = datetime.now(timezone.utc).isoformat()
        self._status    = self.STATUS_PENDING
        self._phase     = ""
        self._frames    = 0
        self._planned   = 0
        self._errors    = 0
        self._message   = ""
        self._error_type: str | None = None
        self._write()

    # ── Public API ────────────────────────────────────────────────────

    def start(self, planned_frames: int) -> None:
        self._status  = self.STATUS_RUNNING
        self._planned = planned_frames
        self._write()

    def update(self, frames_processed: int, phase: str = "") -> None:
        self._frames = frames_processed
        self._phase  = phase
        self._write()

    def record_error(self, message: str = "") -> None:
        self._errors += 1
        if message:
            self._message = message

    def finish_success(self) -> None:
        if self._planned > 0 and self._frames < self._planned * 0.9:
            self._status  = self.STATUS_PARTIAL
            self._message = (
                f"Only {self._frames}/{self._planned} frames were processed "
                f"({self._frames / max(self._planned, 1) * 100:.0f}%)."
            )
        else:
            self._status = self.STATUS_SUCCESS
        self._phase = ""
        self._write()

    def finish_failure(self, exc: BaseException) -> None:
        self._status     = self.STATUS_FAILURE
        self._error_type = type(exc).__name__
        self._message    = _user_message(exc)
        self._phase      = ""
        self._write()

    # ── Helpers ───────────────────────────────────────────────────────

    def _payload(self) -> dict[str, Any]:
        return {
            "run_id":           self._run_id,
            "status":           self._status,
            "started_at":       self._started,
            "updated_at":       datetime.now(timezone.utc).isoformat(),
            "current_phase":    self._phase,
            "frames_processed": self._frames,
            "frames_planned":   self._planned,
            "frame_errors":     self._errors,
            "message":          self._message,
            "error_type":       self._error_type,
        }

    def _write(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._payload(), indent=2, default=str),
                encoding="utf-8",
            )
        except OSError:
            pass   # never crash the pipeline over a status file I/O error


def read_job_status(output_dir: str) -> dict[str, Any] | None:
    """
    Read the current job_status.json from output_dir.

    Returns None if the file does not exist or cannot be parsed.
    """
    path = Path(output_dir) / "job_status.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _user_message(exc: BaseException) -> str:
    """Extract a user-friendly message from an exception."""
    try:
        # PipelineError subclasses expose .user_message()
        return exc.user_message()   # type: ignore[attr-defined]
    except AttributeError:
        return str(exc)
