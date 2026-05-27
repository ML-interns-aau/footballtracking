from __future__ import annotations
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
def write_run_summary(
    *,
    run_id:      str,
    status:      str,
    input_path:  str,
    output_dir:  str,
    config:      dict,
    metrics:     dict,
    result:      dict | None   = None,
    exc:         BaseException | None = None,
    started_at:  str           = "",
    log_file:    str           = "",
) -> Path:
    finished_at = datetime.now(timezone.utc).isoformat()
    failure_trace: str | None = None
    if exc is not None:
        failure_trace = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        if status == "success":
            status = "failure"
    summary: dict[str, Any] = {
        "run_id":        run_id,
        "status":        status,
        "started_at":    started_at or finished_at,
        "finished_at":   finished_at,
        "log_file":      log_file,
        "input": {
            "path": input_path,
        },
        "output": {
            "dir":   output_dir,
            "files": _list_outputs(output_dir),
        },
        "config":  config,
        "metrics": metrics,
        "result":  result or {},
    }
    if failure_trace:
        summary["failure_trace"] = failure_trace
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    history_path = _LOG_DIR / "run_history.jsonl"
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, default=str) + "\n")
    return summary_path
def _list_outputs(output_dir: str) -> list[str]:
    try:
        p = Path(output_dir)
        return sorted(f.name for f in p.iterdir() if f.is_file())
    except Exception:
        return []