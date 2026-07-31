"""Orchestrator: validate a pipeline run's events against StatsBomb ground truth.

A single run is described by a :class:`RunSpec` (or an entry in a
``validation_matches.json`` config). The harness:

1. loads our ``events.json`` and shifts it onto the match clock via
   ``clip_start_s``;
2. loads the StatsBomb events for the corresponding match;
3. matches them per type and computes metrics;
4. writes ``event_validation.json`` and ``event_validation.md`` into the run
   directory and returns the :class:`ValidationResult`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .matching import ValidationResult, validate_events
from .model import SCORED_TYPES
from .ours import load_our_events
from .report import render_markdown, result_to_dict
from .statsbomb import fetch_statsbomb_events, load_statsbomb_events


@dataclass
class RunSpec:
    """Everything needed to validate one clip against one StatsBomb match.

    Parameters
    ----------
    name:
        Human-readable label for the run / report.
    run_dir:
        Pipeline output directory containing ``events.json``.
    clip_start_s:
        Match-clock time (seconds) at which the clip's first frame occurs –
        the kickoff offset the user supplies.
    statsbomb_events:
        Path to a StatsBomb ``events/<match_id>.json`` file (offline path).
    statsbomb_match_id:
        Match id to fetch via ``statsbombpy`` when no local file is given.
    tolerance_s:
        Time window for considering two events the same.
    period:
        Restrict StatsBomb events to this match period, if the clip is within one.
    team_map:
        StatsBomb-team -> ``home``/``away`` mapping for team-aware scoring.
    match_teams:
        Whether team identity must agree for a match (off by default because
        our team-0/1 assignment is arbitrary).
    """

    name: str
    run_dir: str
    clip_start_s: float = 0.0
    statsbomb_events: Optional[str] = None
    statsbomb_match_id: Optional[int] = None
    tolerance_s: float = 3.0
    period: Optional[int] = None
    team_map: dict[str, str] = field(default_factory=dict)
    match_teams: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunSpec":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown RunSpec field(s): {sorted(unknown)}")
        return cls(**d)


def validate_run(spec: RunSpec, write_reports: bool = True) -> ValidationResult:
    """Validate a single run and optionally write report files into its dir."""
    run_dir = Path(spec.run_dir)

    ours = load_our_events(run_dir, clip_start_s=spec.clip_start_s)

    if spec.statsbomb_events:
        truth = load_statsbomb_events(
            spec.statsbomb_events, spec.team_map, period=spec.period
        )
    elif spec.statsbomb_match_id is not None:
        truth = fetch_statsbomb_events(
            spec.statsbomb_match_id, spec.team_map, period=spec.period
        )
    else:
        raise ValueError(
            f"Run '{spec.name}': provide either 'statsbomb_events' (file) or "
            "'statsbomb_match_id' (statsbombpy fetch)."
        )

    result = validate_events(
        ours,
        truth,
        run_name=spec.name,
        tolerance_s=spec.tolerance_s,
        match_teams=spec.match_teams,
        types=SCORED_TYPES,
    )

    if write_reports:
        json_path = run_dir / "event_validation.json"
        md_path = run_dir / "event_validation.md"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_to_dict(result), f, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(render_markdown(result))

    return result


def validate_from_config(config_path: str | Path) -> list[ValidationResult]:
    """Validate every run listed in a ``validation_matches.json`` config."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    runs = config.get("runs", [])
    if not runs:
        raise ValueError(f"No 'runs' array found in {config_path}")
    return [validate_run(RunSpec.from_dict(r)) for r in runs]
