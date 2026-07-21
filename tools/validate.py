"""Definition-of-done for the corner-kick timing rebuild (see
IMPLEMENTATION_PLAN_FINAL.md §7): runs the fully automatic pipeline
(no --kick_frame/--contact_frame overrides) on clip_1 and clip_2 with a
single shared threshold set, compares the auto-picked kick/contact frames
against ground truth, and exits non-zero if either pick misses by more than
±3 frames on either clip.

Ground truth lives ONLY here — never hardcode it into detection logic.

Usage:
    python tools/validate.py
    python tools/validate.py --clip1 /path/to/clip_1.mp4 --clip2 /path/to/clip_2.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.extract_corner_snapshots import parse_args as tool_parse_args, run as tool_run

REPO_ROOT = Path(__file__).resolve().parent.parent

GROUND_TRUTH = {
    "clip_1": {"path": REPO_ROOT / "clip_1.mp4", "kick_frame": 95, "contact_frame": 145},
    "clip_2": {"path": REPO_ROOT / "clip_2.mp4", "kick_frame": 69, "contact_frame": 112},
}

TOLERANCE_FRAMES = 3


def validate_clip(name: str, video_path: Path, gt_kick: int, gt_contact: int, extra_args: list[str]) -> dict:
    args = tool_parse_args(["--input", str(video_path), *extra_args])
    result = tool_run(args)

    kick_frame = result["kick_frame"]
    contact_frame = result["contact_frame"]

    kick_err = abs(kick_frame - gt_kick)
    contact_err = None if contact_frame is None else abs(contact_frame - gt_contact)

    kick_ok = kick_err <= TOLERANCE_FRAMES
    contact_ok = contact_frame is not None and contact_err <= TOLERANCE_FRAMES

    print(f"\n[{name}] kick:    auto={kick_frame:>4}  gt={gt_kick:>4}  err={kick_err:>3}  "
          f"{'PASS' if kick_ok else 'FAIL'}  (confidence={result['kick_confidence']})")
    if contact_frame is None:
        print(f"[{name}] contact: auto=None  gt={gt_contact:>4}  err=N/A  FAIL (no contact frame detected)")
    else:
        print(f"[{name}] contact: auto={contact_frame:>4}  gt={gt_contact:>4}  err={contact_err:>3}  "
              f"{'PASS' if contact_ok else 'FAIL'}  (confidence={result['contact_confidence']})")

    n_decoys = len(result["decoy_ids"])
    print(f"[{name}] rejected {n_decoys} static decoy cluster(s); gate_stats={result['gate_stats']}")

    return {
        "name": name, "kick_ok": kick_ok, "contact_ok": contact_ok,
        "kick_err": kick_err, "contact_err": contact_err,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip1", default=None, help="Override path to clip_1.mp4")
    p.add_argument("--clip2", default=None, help="Override path to clip_2.mp4")
    p.add_argument("extra", nargs=argparse.REMAINDER,
                    help="Extra args forwarded verbatim to extract_corner_snapshots (after --).")
    args = p.parse_args(argv)

    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    overrides = {"clip_1": args.clip1, "clip_2": args.clip2}
    all_ok = True
    results = []
    for name, gt in GROUND_TRUTH.items():
        video_path = Path(overrides[name]) if overrides[name] else gt["path"]
        if not video_path.exists():
            print(f"[{name}] SKIPPED: {video_path} not found", file=sys.stderr)
            all_ok = False
            continue
        r = validate_clip(name, video_path, gt["kick_frame"], gt["contact_frame"], extra)
        results.append(r)
        all_ok = all_ok and r["kick_ok"] and r["contact_ok"]

    print("\n" + "=" * 60)
    if all_ok and results:
        print(f"ALL CLIPS PASSED (±{TOLERANCE_FRAMES} frame tolerance)")
    else:
        print(f"VALIDATION FAILED (tolerance = ±{TOLERANCE_FRAMES} frames)")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
