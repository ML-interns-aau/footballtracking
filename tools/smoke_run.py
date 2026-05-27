from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MAIN_PY      = PROJECT_ROOT / "main.py"
REQUIRED_ARTIFACTS = ("analytics.json", "events.json")
EXIT_OK              = 0
EXIT_USAGE           = 1
EXIT_PIPELINE_FAILED = 2
EXIT_MISSING         = 3
EXIT_EMPTY           = 4
def _log(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)
def _build_command(args: argparse.Namespace, output_dir: Path) -> List[str]:
    cmd = [
        args.python,
        str(MAIN_PY),
        "--input",      str(args.input),
        "--output_dir", str(output_dir),
        "--max_frames", str(args.max_frames),
    ]
    if args.extra:
        cmd.extend(args.extra)
    return cmd
def _run_pipeline(cmd: List[str], timeout: int) -> int:
    _log("running pipeline: " + " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _log(f"FAIL: pipeline timed out after {timeout}s")
        return EXIT_PIPELINE_FAILED
    except FileNotFoundError as exc:
        _log(f"FAIL: could not launch subprocess ({exc}). Is '{cmd[0]}' on PATH?")
        return EXIT_PIPELINE_FAILED
    if result.returncode != 0:
        _log(f"FAIL: pipeline exited with code {result.returncode}")
        return EXIT_PIPELINE_FAILED
    _log("pipeline finished cleanly")
    return EXIT_OK
def _verify_artifact(path: Path, require_non_empty: bool) -> Optional[str]:
    if not path.exists():
        return f"missing: {path}"
    if path.stat().st_size == 0:
        return f"empty file: {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"invalid JSON in {path}: {exc}"
    if require_non_empty and (data == [] or data == {} or data is None):
        return f"empty JSON payload in {path}"
    return None
def _verify_artifacts(output_dir: Path, require_events: bool) -> int:
    _log(f"verifying artifacts in {output_dir}")
    err = _verify_artifact(output_dir / "analytics.json", require_non_empty=True)
    if err:
        _log(f"FAIL ({REQUIRED_ARTIFACTS[0]}): {err}")
        return EXIT_EMPTY if "empty" in err or "invalid" in err else EXIT_MISSING
    err = _verify_artifact(output_dir / "events.json", require_non_empty=require_events)
    if err:
        _log(f"FAIL ({REQUIRED_ARTIFACTS[1]}): {err}")
        return EXIT_EMPTY if "empty" in err or "invalid" in err else EXIT_MISSING
    _log("OK: analytics.json and events.json present and valid")
    return EXIT_OK
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="smoke_run",
        description="Black-box smoke test for the football tracking pipeline.",
    )
    parser.add_argument("--input",       required=True,
                        help="Path to the input video file forwarded to main.py.")
    parser.add_argument("--max-frames",  type=int, default=30,
                        help="Number of frames to process (default: 30, small for speed).")
    parser.add_argument("--output-dir",  type=str, default=None,
                        help="Where main.py writes its outputs. If omitted, a temp "
                             "directory is created and cleaned up after the run.")
    parser.add_argument("--python",      type=str, default=sys.executable,
                        help="Python interpreter used to launch main.py (default: current).")
    parser.add_argument("--timeout",     type=int, default=300,
                        help="Subprocess timeout in seconds (default: 300).")
    parser.add_argument("--require-events", action="store_true",
                        help="Treat an empty events.json payload as a failure. "
                             "Off by default since short clips may produce no events.")
    parser.add_argument("--keep-output", action="store_true",
                        help="Don't delete the temp output dir on success. Useful for inspection.")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="Extra args after '--' are forwarded to main.py verbatim. "
                             "Example: tools/smoke_run.py --input v.mp4 -- --conf 0.3")
    args = parser.parse_args(argv)
    if not MAIN_PY.exists():
        _log(f"FAIL: main.py not found at {MAIN_PY}")
        return EXIT_USAGE
    input_path = Path(args.input)
    if not input_path.exists():
        _log(f"FAIL: input not found: {input_path}")
        return EXIT_USAGE
    if args.max_frames <= 0:
        _log(f"FAIL: --max-frames must be > 0 (got {args.max_frames})")
        return EXIT_USAGE
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    tmp_root: Optional[str] = None
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_root  = tempfile.mkdtemp(prefix="football_smoke_")
        output_dir = Path(tmp_root)
        _log(f"using temp output dir: {output_dir}")
    try:
        cmd  = _build_command(args, output_dir)
        rc   = _run_pipeline(cmd, args.timeout)
        if rc != EXIT_OK:
            return rc
        rc = _verify_artifacts(output_dir, require_events=args.require_events)
        return rc
    finally:
        if tmp_root and not args.keep_output:
            shutil.rmtree(tmp_root, ignore_errors=True)
if __name__ == "__main__":
    raise SystemExit(main())