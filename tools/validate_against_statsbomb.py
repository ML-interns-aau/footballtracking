"""CLI: validate pipeline event detection against StatsBomb ground truth.

Two modes
---------
Single run (flags):
    python tools/validate_against_statsbomb.py \
        --run-dir data/insights/England_preprocessed_20260625_174216 \
        --statsbomb-events data/statsbomb/events/3795506.json \
        --clip-start-s 0 --period 1 --tolerance-s 3.0 --name England_clip

Batch (config file):
    python tools/validate_against_statsbomb.py --config configs/validation_matches.json

Reports (event_validation.json / .md) are written into each run directory, and
a summary table is printed to stdout. Exit code is non-zero on error so the
command can gate CI / scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (python tools/validate_against_statsbomb.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation.events import RunSpec, validate_from_config  # noqa: E402
from src.validation.events.harness import validate_run  # noqa: E402
from src.validation.events.matching import ValidationResult  # noqa: E402


def _print_summary(result: ValidationResult) -> None:
    print(f"\n=== {result.run_name} "
          f"(+/-{result.tolerance_s:g}s, teams={'on' if result.match_teams else 'off'}) ===")
    header = f"{'type':<18} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}"
    print(header)
    print("-" * len(header))
    for t, r in result.per_type.items():
        print(f"{t:<18} {r.tp:>4} {r.fp:>4} {r.fn:>4} "
              f"{r.precision:>7.3f} {r.recall:>7.3f} {r.f1:>7.3f}")
    print("-" * len(header))
    print(f"{'OVERALL':<18} {result.overall_tp:>4} {result.overall_fp:>4} "
          f"{result.overall_fn:>4} {result.overall_precision:>7.3f} "
          f"{result.overall_recall:>7.3f} {result.overall_f1:>7.3f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate event detection vs StatsBomb.")
    p.add_argument("--config", type=str, help="Path to validation_matches.json for batch mode.")
    p.add_argument("--run-dir", type=str, help="Pipeline run dir containing events.json.")
    p.add_argument("--statsbomb-events", type=str, help="Path to StatsBomb events/<match_id>.json.")
    p.add_argument("--statsbomb-match-id", type=int, help="Match id (requires statsbombpy).")
    p.add_argument("--clip-start-s", type=float, default=0.0, help="Kickoff offset of clip (s).")
    p.add_argument("--period", type=int, default=None, choices=[1, 2], help="Restrict to a half.")
    p.add_argument("--tolerance-s", type=float, default=3.0, help="Match time window (s).")
    p.add_argument("--match-teams", action="store_true", help="Require team agreement.")
    p.add_argument("--name", type=str, default=None, help="Run label for reports.")
    args = p.parse_args(argv)

    try:
        if args.config:
            results = validate_from_config(args.config)
        else:
            if not args.run_dir:
                p.error("provide --config, or --run-dir with a StatsBomb source")
            if not (args.statsbomb_events or args.statsbomb_match_id):
                p.error("provide --statsbomb-events or --statsbomb-match-id")
            spec = RunSpec(
                name=args.name or Path(args.run_dir).name,
                run_dir=args.run_dir,
                clip_start_s=args.clip_start_s,
                statsbomb_events=args.statsbomb_events,
                statsbomb_match_id=args.statsbomb_match_id,
                tolerance_s=args.tolerance_s,
                period=args.period,
                match_teams=args.match_teams,
            )
            results = [validate_run(spec)]
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"[validation error] {exc}", file=sys.stderr)
        return 1

    for result in results:
        _print_summary(result)
    print(f"\nReports written into each run directory "
          f"(event_validation.json / event_validation.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
