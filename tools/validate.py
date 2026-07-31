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
    "testClip": {"path": REPO_ROOT / "testClip.mp4", "kick_frame": 60, "contact_frame": 112},
}

INFORMATIONAL_CLIPS = {
    "clip_3": REPO_ROOT / "clip_3.mp4",
    "clip_04": REPO_ROOT / "clip 04.mp4",
}

TOLERANCE_FRAMES = 3


def validate_clip(name: str, video_path: Path, gt_kick: int, gt_contact: int, extra_args: list[str]) -> dict:
    """Tier-aware validation (see CONTACT_DETECTION_SPEC.md): a clip whose true
    contact is genuinely undetectable can never hit ±3 frames, so the bar is
    tier-conditional --
      Tier 1/2 (exact frame reported): must be within TOLERANCE_FRAMES of
        ground truth. A wrong Tier 1/2 frame is a HARD failure -- a confident
        wrong answer is worse than an honest abstain and must fail loudly.
      Tier 3 (abstain, window reported): passes if the reported window
        CONTAINS the ground-truth contact frame.
    Kick detection keeps the flat ±3 bar (it doesn't have a not-guessing tier).
    """
    args = tool_parse_args(["--input", str(video_path), *extra_args])
    result = tool_run(args)

    kick_frame = result["kick_frame"]
    contact_frame = result["contact_frame"]
    contact_tier = result["contact_tier"]
    contact_window = result["contact_window"]

    kick_err = abs(kick_frame - gt_kick)
    kick_ok = kick_err <= TOLERANCE_FRAMES
    print(f"\n[{name}] kick:    auto={kick_frame:>4}  gt={gt_kick:>4}  err={kick_err:>3}  "
          f"{'PASS' if kick_ok else 'FAIL'}  (confidence={result['kick_confidence']})")

    contact_err = None
    hard_fail = False
    if contact_tier in (1, 2) and contact_frame is not None:
        contact_err = abs(contact_frame - gt_contact)
        contact_ok = contact_err <= TOLERANCE_FRAMES
        if not contact_ok:
            hard_fail = True
        print(f"[{name}] contact: tier={contact_tier}  auto={contact_frame:>4}  gt={gt_contact:>4}  err={contact_err:>3}  "
              f"{'PASS' if contact_ok else 'HARD FAIL (confident wrong frame)'}  (confidence={result['contact_confidence']})")
    elif contact_tier == 3:
        if contact_window is not None:
            w_lo, w_hi = contact_window
            contact_ok = w_lo <= gt_contact <= w_hi
            print(f"[{name}] contact: tier=3 (abstain)  window=[{w_lo},{w_hi}]  gt={gt_contact}  "
                  f"{'PASS (window contains gt)' if contact_ok else 'FAIL (window misses gt)'}")
        else:
            contact_ok = False
            print(f"[{name}] contact: tier=3 (abstain), no window reported -- FAIL")
    else:
        contact_ok = False
        print(f"[{name}] contact: auto=None, no tier/window reported -- FAIL")

    n_decoys = len(result["decoy_ids"])
    print(f"[{name}] rejected {n_decoys} static decoy cluster(s); gate_stats={result['gate_stats']}")

    return {
        "name": name, "kick_ok": kick_ok, "contact_ok": contact_ok,
        "kick_err": kick_err, "contact_err": contact_err, "hard_fail": hard_fail,
        "contact_tier": contact_tier,
    }


def report_informational(name: str, video_path: Path, extra_args: list[str]) -> None:
    """Runs a clip with no confirmed ground truth -- reports the auto-picked
    frames for manual spot-checking without gating pass/fail on them."""
    args = tool_parse_args(["--input", str(video_path), *extra_args])
    result = tool_run(args)
    kick_frame, contact_frame = result["kick_frame"], result["contact_frame"]
    print(f"\n[{name}] (informational, no confirmed ground truth) "
          f"kick={kick_frame} (confidence={result['kick_confidence']})  "
          f"contact={contact_frame} (confidence={result['contact_confidence']})")


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
    any_hard_fail = False
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
        any_hard_fail = any_hard_fail or r["hard_fail"]

    for name, video_path in INFORMATIONAL_CLIPS.items():
        if not video_path.exists():
            print(f"[{name}] SKIPPED: {video_path} not found", file=sys.stderr)
            continue
        report_informational(name, video_path, extra)

    print("\n" + "=" * 60)
    if any_hard_fail:
        print("VALIDATION FAILED -- at least one clip emitted a confident WRONG exact contact frame "
              "(Tier 1/2 outside tolerance). This is the worst-case failure mode and fails the whole run.")
    elif all_ok and results:
        print(f"ALL CLIPS PASSED (tier-aware: Tier 1/2 within ±{TOLERANCE_FRAMES} frames, "
              f"Tier 3 window contains ground truth)")
    else:
        print("VALIDATION FAILED")
    print("=" * 60)

    return 0 if (all_ok and not any_hard_fail) else 1


if __name__ == "__main__":
    raise SystemExit(main())
