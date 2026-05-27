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
    def format(self, record: logging.LogRecord) -> str:
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
    _COLORS = {
        "DEBUG":    "\033[36m",
        "INFO":     "\033[32m",
        "WARNING":  "\033[33m",
        "ERROR":    "\033[31m",
        "CRITICAL": "\033[35m",
    }
    _RESET = "\033[0m"
    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelname, "")
        ts    = datetime.now(timezone.utc).strftime("%H:%M:%S")
        base  = f"{color}[{record.levelname[0]}]{self._RESET} {ts} {record.name} — {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base
def setup_logging(run_id: str = "", log_dir: Path | None = None) -> Path:
    global _CONFIGURED
    log_dir = Path(log_dir) if log_dir else _LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id    = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_file  = log_dir / f"run_{run_id}.log"
    err_file  = log_dir / "errors.log"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if _CONFIGURED:
        root.handlers.clear()
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JSONFormatter())
    root.addHandler(fh)
    eh = logging.handlers.RotatingFileHandler(
        err_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    eh.setLevel(logging.WARNING)
    eh.setFormatter(_JSONFormatter())
    root.addHandler(eh)
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
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)