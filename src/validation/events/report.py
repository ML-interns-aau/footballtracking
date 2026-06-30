"""Render :class:`ValidationResult` objects into JSON and Markdown reports.

The Markdown report is designed for the manual "compare every detected event
with the actual video" step: it lists each false positive (a detection to
verify is spurious) and each false negative (a missed event) with its match
clock, so a reviewer can jump straight to that timestamp in the clip.
"""

from __future__ import annotations

from typing import Any

from .matching import TypeResult, ValidationResult


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def result_to_dict(result: ValidationResult) -> dict[str, Any]:
    """Serialise a :class:`ValidationResult` to a JSON-friendly dict."""

    def type_dict(r: TypeResult) -> dict[str, Any]:
        return {
            "tp": r.tp,
            "fp": r.fp,
            "fn": r.fn,
            "precision": _round(r.precision),
            "recall": _round(r.recall),
            "f1": _round(r.f1),
            "mean_time_error_s": _round(r.mean_time_error_s),
            "mean_spatial_error_m": _round(r.mean_spatial_error_m),
            "team_agreement_rate": _round(r.team_agreement_rate),
            "false_positives": [
                {"clock": e.clock, "match_time_s": _round(e.match_time_s, 2), "team": e.team}
                for e in r.false_positives
            ],
            "false_negatives": [
                {"clock": e.clock, "match_time_s": _round(e.match_time_s, 2), "team": e.team}
                for e in r.false_negatives
            ],
        }

    return {
        "run_name": result.run_name,
        "tolerance_s": result.tolerance_s,
        "match_teams": result.match_teams,
        "overall": {
            "tp": result.overall_tp,
            "fp": result.overall_fp,
            "fn": result.overall_fn,
            "precision": _round(result.overall_precision),
            "recall": _round(result.overall_recall),
            "f1": _round(result.overall_f1),
        },
        "per_type": {t: type_dict(r) for t, r in result.per_type.items()},
    }


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_markdown(result: ValidationResult) -> str:
    """Render a human-readable Markdown validation report."""
    lines: list[str] = []
    lines.append(f"# Event validation report — {result.run_name}")
    lines.append("")
    lines.append(
        f"Tolerance: ±{result.tolerance_s:g}s · "
        f"Team-aware matching: {'on' if result.match_teams else 'off'}"
    )
    lines.append("")
    lines.append("## Summary (scored types)")
    lines.append("")
    lines.append("| Type | TP | FP | FN | Precision | Recall | F1 | mean Δt (s) | mean Δxy (m) |")
    lines.append("|------|----|----|----|-----------|--------|----|-------------|--------------|")
    for t, r in result.per_type.items():
        lines.append(
            f"| {t} | {r.tp} | {r.fp} | {r.fn} | "
            f"{_pct(r.precision)} | {_pct(r.recall)} | {_pct(r.f1)} | "
            f"{_pct(r.mean_time_error_s)} | {_pct(r.mean_spatial_error_m)} |"
        )
    lines.append(
        f"| **overall** | {result.overall_tp} | {result.overall_fp} | {result.overall_fn} | "
        f"**{_pct(result.overall_precision)}** | **{_pct(result.overall_recall)}** | "
        f"**{_pct(result.overall_f1)}** | - | - |"
    )
    lines.append("")

    # Detailed review lists for the manual video-comparison step.
    for t, r in result.per_type.items():
        if not r.false_positives and not r.false_negatives:
            continue
        lines.append(f"## {t} — items to review")
        lines.append("")
        if r.false_positives:
            lines.append("**False positives** (our detections to verify against video):")
            for e in r.false_positives:
                team = f" [{e.team}]" if e.team else ""
                lines.append(f"- `{e.clock}` (t={e.match_time_s:.2f}s){team}")
            lines.append("")
        if r.false_negatives:
            lines.append("**False negatives** (StatsBomb events we missed):")
            for e in r.false_negatives:
                team = f" [{e.team}]" if e.team else ""
                lines.append(f"- `{e.clock}` (t={e.match_time_s:.2f}s){team}")
            lines.append("")

    return "\n".join(lines)
