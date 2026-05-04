"""
Structured logging setup for the Football Analytics Pipeline.

Provides a single `setup_logging()` call that configures:
  - A rotating file handler  → logs/<run_id>.log
  - A console handler        → stdout (JSON-like format)
  - A separate error file    → logs/errors.log

Call `get_logger(__name__)` in any module to get a pre-configured logger.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_CONFIGURED = False


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            payload.update(record.extra)
        return json.dumps(payload, default=str)


class _HumanFormatter(logging.Formatter):
    """Human-readable console format with colour coding."""

    _COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        color = self._COLORS.get(record.levelname, "")
        ts    = datetime.now(timezone.utc).strftime("%H:%M:%S")
        base  = f"{color}[{record.levelname[0]}]{self._RESET} {ts} {record.name} — {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(run_id: str = "", log_dir: Path | None = None) -> Path:
    """
    Configure root logger with file (JSON) + console (human) handlers.

    Parameters
    ----------
    run_id  : Unique identifier for the run (timestamp used if empty).
    log_dir : Override default logs/ directory.

    Returns
    -------
    Path to the per-run log file.
    """
    global _CONFIGURED

    log_dir = Path(log_dir) if log_dir else _LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id    = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_file  = log_dir / f"run_{run_id}.log"
    err_file  = log_dir / "errors.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if _CONFIGURED:
        # Avoid duplicate handlers on hot-reload (e.g. Streamlit)
        root.handlers.clear()

    # ── Per-run JSON rotating file ─────────────────────────────────────────
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JSONFormatter())
    root.addHandler(fh)

    # ── Persistent error file (all runs) ──────────────────────────────────
    eh = logging.handlers.RotatingFileHandler(
        err_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    eh.setLevel(logging.WARNING)
    eh.setFormatter(_JSONFormatter())
    root.addHandler(eh)

    # ── Console (stdout) — human-readable ────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_HumanFormatter())
    root.addHandler(ch)

    _CONFIGURED = True

    logging.getLogger(__name__).info(
        "Logging initialised",
        extra={"extra": {"run_id": run_id, "log_file": str(log_file)}},
    )
    return log_file


def get_logger(name: str) -> logging.Logger:
    """Return a logger for *name*, setting up logging if not already done."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
