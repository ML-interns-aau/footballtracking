"""Load StatsBomb open-data events and normalise them into :class:`NormEvent`.

The harness is offline-first: it reads a StatsBomb ``events/<match_id>.json``
file straight from disk (the format published in the StatsBomb open-data
repository).  If the optional ``statsbombpy`` package is installed, the events
can also be fetched by match id via :func:`fetch_statsbomb_events`.

StatsBomb -> canonical type mapping
-----------------------------------
``pass``          StatsBomb "Pass", completed (no ``pass.outcome``), not a cross
``cross``         StatsBomb "Pass" with ``pass.cross == True``
``interception``  StatsBomb "Interception" (kept regardless of outcome)
``recovery``      StatsBomb "Ball Recovery"
``switch_of_play``StatsBomb "Pass" with ``pass.switch == True`` (diagnostic)
``skill_move``    StatsBomb "Dribble" (diagnostic)

This mirrors how our detector emits events: a cross is its own event (not also
counted as a pass), interceptions replace the lost pass, etc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .model import NormEvent, statsbomb_to_pitch


def _match_time_s(ev: dict[str, Any]) -> float:
    """Absolute match-clock seconds for a StatsBomb event.

    StatsBomb ``minute`` is cumulative across periods (period-2 events have
    ``minute >= 45``), so ``minute * 60 + second`` is a single continuous
    timeline that lines up with our clip clock once the kickoff offset is known.
    """
    return float(ev.get("minute", 0)) * 60.0 + float(ev.get("second", 0))


def _team_side(ev: dict[str, Any], team_map: dict[str, str]) -> Optional[str]:
    """Resolve a StatsBomb team to our canonical ``home``/``away`` side.

    ``team_map`` maps StatsBomb team *name* (and/or stringified id) to a side.
    Returns ``None`` when the team is not in the map (team-aware matching then
    simply treats the side as unknown).
    """
    team = ev.get("team") or {}
    name = team.get("name")
    tid = team.get("id")
    if name is not None and name in team_map:
        return team_map[name]
    if tid is not None and str(tid) in team_map:
        return team_map[str(tid)]
    return None


def _location(ev: dict[str, Any], pw: float, ph: float) -> tuple[Optional[float], Optional[float]]:
    loc = ev.get("location")
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        return statsbomb_to_pitch(float(loc[0]), float(loc[1]), pw, ph)
    return None, None


def normalize_statsbomb_events(
    raw_events: list[dict[str, Any]],
    team_map: Optional[dict[str, str]] = None,
    *,
    pitch_width_m: float = 105.0,
    pitch_height_m: float = 68.0,
    period: Optional[int] = None,
) -> list[NormEvent]:
    """Convert raw StatsBomb event dicts into canonical :class:`NormEvent` records.

    Parameters
    ----------
    raw_events:
        Parsed StatsBomb events array.
    team_map:
        Optional StatsBomb-team -> ``home``/``away`` mapping for team-aware
        scoring.
    period:
        If given, keep only events from this match period (1 or 2). Useful when
        a clip is known to fall entirely within one half.
    """
    team_map = team_map or {}
    out: list[NormEvent] = []

    for ev in raw_events:
        if period is not None and ev.get("period") != period:
            continue

        type_name = (ev.get("type") or {}).get("name")
        canonical: Optional[str] = None

        if type_name == "Pass":
            pass_obj = ev.get("pass") or {}
            completed = "outcome" not in pass_obj  # StatsBomb: outcome present => not completed
            if pass_obj.get("cross"):
                canonical = "cross"
            elif pass_obj.get("switch") and completed:
                # A switch is also a completed pass; emit both so it scores
                # against our pass detector and our switch_of_play detector.
                x, y = _location(ev, pitch_width_m, pitch_height_m)
                side = _team_side(ev, team_map)
                t = _match_time_s(ev)
                out.append(NormEvent("pass", t, side, x, y, "statsbomb", ev))
                canonical = "switch_of_play"
            elif completed:
                canonical = "pass"
        elif type_name == "Interception":
            canonical = "interception"
        elif type_name == "Ball Recovery":
            canonical = "recovery"
        elif type_name == "Dribble":
            canonical = "skill_move"

        if canonical is None:
            continue

        x, y = _location(ev, pitch_width_m, pitch_height_m)
        out.append(
            NormEvent(
                type=canonical,
                match_time_s=_match_time_s(ev),
                team=_team_side(ev, team_map),
                x=x,
                y=y,
                source="statsbomb",
                raw=ev,
            )
        )

    out.sort(key=lambda e: e.match_time_s)
    return out


def load_statsbomb_events(
    path: str | Path,
    team_map: Optional[dict[str, str]] = None,
    **kwargs: Any,
) -> list[NormEvent]:
    """Load and normalise a StatsBomb ``events/<match_id>.json`` file from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"StatsBomb events file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a StatsBomb events array (list), got {type(raw).__name__} in {path}"
        )
    return normalize_statsbomb_events(raw, team_map, **kwargs)


def fetch_statsbomb_events(
    match_id: int,
    team_map: Optional[dict[str, str]] = None,
    **kwargs: Any,
) -> list[NormEvent]:
    """Fetch events for a match via the optional ``statsbombpy`` package.

    Raises a clear error if ``statsbombpy`` is not installed; the disk-based
    :func:`load_statsbomb_events` is the supported default path.
    """
    try:
        from statsbombpy import sb  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "fetch_statsbomb_events requires the optional 'statsbombpy' package "
            "(pip install statsbombpy). Alternatively download the match's "
            "events JSON and use load_statsbomb_events()."
        ) from exc

    df = sb.events(match_id=match_id)
    raw = df.to_dict(orient="records")
    return normalize_statsbomb_events(raw, team_map, **kwargs)
