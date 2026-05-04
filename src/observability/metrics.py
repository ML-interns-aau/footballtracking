"""
Pipeline timing metrics collector.

Usage
-----
    metrics = PipelineMetrics()

    with metrics.phase("detection"):
        detections = detector.detect(frame)

    metrics.tick()   # call once per frame

    summary = metrics.summary()
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List

from contextlib import contextmanager


@dataclass
class _PhaseStats:
    calls:      int   = 0
    total_s:    float = 0.0
    min_s:      float = float("inf")
    max_s:      float = 0.0

    def record(self, elapsed: float) -> None:
        self.calls   += 1
        self.total_s += elapsed
        self.min_s    = min(self.min_s, elapsed)
        self.max_s    = max(self.max_s, elapsed)

    @property
    def avg_s(self) -> float:
        return self.total_s / self.calls if self.calls else 0.0

    def to_dict(self) -> dict:
        return {
            "calls":    self.calls,
            "total_ms": round(self.total_s * 1000, 2),
            "avg_ms":   round(self.avg_s * 1000, 2),
            "min_ms":   round(self.min_s * 1000, 2) if self.calls else 0,
            "max_ms":   round(self.max_s * 1000, 2),
        }


class PipelineMetrics:
    """
    Lightweight timer/counter for pipeline phases.

    Thread-safety: single-threaded use only (main pipeline loop).
    """

    def __init__(self) -> None:
        self._phases: Dict[str, _PhaseStats] = defaultdict(_PhaseStats)
        self._frame_times: List[float]        = []
        self._pipeline_start: float           = time.perf_counter()
        self._last_frame_ts: float            = self._pipeline_start
        self._frames_processed: int           = 0
        self._frames_skipped: int             = 0
        self._detection_counts: List[int]     = []
        self._errors: int                     = 0

    # ── Context manager ───────────────────────────────────────────────
    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time a named pipeline phase."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._phases[name].record(time.perf_counter() - t0)

    # ── Per-frame counters ────────────────────────────────────────────
    def tick(self, n_detections: int = 0) -> None:
        """Call once per processed frame."""
        now = time.perf_counter()
        self._frame_times.append(now - self._last_frame_ts)
        self._last_frame_ts = now
        self._frames_processed += 1
        self._detection_counts.append(n_detections)

    def skip(self) -> None:
        """Call when a source frame is dropped (FPS downsampling)."""
        self._frames_skipped += 1

    def record_error(self) -> None:
        """Increment the error counter (call in except blocks)."""
        self._errors += 1

    # ── Summary ───────────────────────────────────────────────────────
    def summary(self) -> dict:
        """Return a JSON-serialisable summary dict."""
        wall_s    = time.perf_counter() - self._pipeline_start
        fps_real  = self._frames_processed / wall_s if wall_s > 0 else 0

        ft = self._frame_times
        avg_frame_ms = (sum(ft) / len(ft) * 1000) if ft else 0
        p95_frame_ms = (sorted(ft)[int(len(ft) * 0.95)] * 1000) if len(ft) >= 20 else avg_frame_ms

        dc = self._detection_counts
        avg_detections = sum(dc) / len(dc) if dc else 0

        return {
            "wall_time_s":       round(wall_s, 2),
            "frames_processed":  self._frames_processed,
            "frames_skipped":    self._frames_skipped,
            "throughput_fps":    round(fps_real, 2),
            "avg_frame_ms":      round(avg_frame_ms, 2),
            "p95_frame_ms":      round(p95_frame_ms, 2),
            "avg_detections":    round(avg_detections, 2),
            "errors":            self._errors,
            "phases":            {k: v.to_dict() for k, v in self._phases.items()},
        }
