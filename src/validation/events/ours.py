"""Load our pipeline's ``events.json`` and normalise it into :class:`NormEvent`.

Our events carry a clip-relative ``timestamp_ms`` (frame / effective_fps).
To compare against StatsBomb's match clock we add ``clip_start_s`` – the match
time at which the clip begins (the user-supplied kickoff offset).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.exporters.output_schema import OutputFiles

from .model import NormEvent


# Position field to use per event type when projecting to pitch space.
_XY_FIELD = {
    "pass": "start_xy",
    "switch_of_play": "start_xy",
    "interception": "start_xy",
    "recovery": "recovery_xy",
    "final_third_entry": "entry_xy",
    "penalty_area_entry": "entry_xy",
}


def _side_from_team_string(team: Optional[str], team_id_map: dict[str, str]) -> Optional[str]:
    """Map an emitted team string (e.g. ``"Team 0"``) to ``home``/``away``.

    Uses the ``team_id_map`` recorded in ``analytics.json`` (defaults to
    ``{0: home, 1: away}``). Returns ``None`` for referees/unknown.
    """
    if not isinstance(team, str) or not team.startswith("Team "):
        return None
    try:
        tid = team.split()[1]
    except (IndexError, ValueError):
        return None
    return team_id_map.get(tid) or team_id_map.get(str(tid))


def _event_team_field(ev: dict[str, Any]) -> Optional[str]:
    """The team string that owns an event, across event schemas."""
    return ev.get("team") or ev.get("passer_team") or ev.get("player_team")


def _event_xy(ev: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    field = _XY_FIELD.get(ev.get("type", ""))
    if field and isinstance(ev.get(field), (list, tuple)) and len(ev[field]) >= 2:
        return float(ev[field][0]), float(ev[field][1])
    # cross stores origin as separate scalar fields
    if ev.get("origin_x_m") is not None and ev.get("origin_y_m") is not None:
        return float(ev["origin_x_m"]), float(ev["origin_y_m"])
    return None, None


def _read_team_id_map(run_dir: Path) -> dict[str, str]:
    """Read ``team_id_map`` from the run's analytics.json, falling back to default."""
    analytics_path = run_dir / OutputFiles.ANALYTICS_JSON
    default = {"0": "home", "1": "away"}
    if not analytics_path.exists():
        return default
    try:
        with open(analytics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = (data.get("match_info") or {}).get("team_id_map") or {}
        # keys may be ints or strings depending on JSON round-trips
        return {str(k): v for k, v in raw.items()} or default
    except (json.JSONDecodeError, OSError):
        return default


def normalize_our_events(
    raw_events: list[dict[str, Any]],
    clip_start_s: float,
    team_id_map: Optional[dict[str, str]] = None,
) -> list[NormEvent]:
    """Convert our raw event dicts to canonical :class:`NormEvent` records."""
    team_id_map = team_id_map or {"0": "home", "1": "away"}
    out: list[NormEvent] = []
    for ev in raw_events:
        etype = ev.get("type")
        if not etype:
            continue
        ts_ms = ev.get("timestamp_ms")
        if ts_ms is None:
            continue
        x, y = _event_xy(ev)
        out.append(
            NormEvent(
                type=etype,
                match_time_s=clip_start_s + float(ts_ms) / 1000.0,
                team=_side_from_team_string(_event_team_field(ev), team_id_map),
                x=x,
                y=y,
                source="ours",
                raw=ev,
            )
        )
    out.sort(key=lambda e: e.match_time_s)
    return out


def load_our_events(
    run_dir: str | Path,
    clip_start_s: float = 0.0,
    team_id_map: Optional[dict[str, str]] = None,
) -> list[NormEvent]:
    """Load and normalise ``events.json`` from a pipeline run directory."""
    run_dir = Path(run_dir)
    events_path = run_dir / "events.json"
    if not events_path.exists():
        raise FileNotFoundError(f"Pipeline events.json not found: {events_path}")
    with open(events_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected events.json to be a JSON array, got {type(raw).__name__}")
    if team_id_map is None:
        team_id_map = _read_team_id_map(run_dir)
    return normalize_our_events(raw, clip_start_s, team_id_map)
