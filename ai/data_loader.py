"""Build a grounded match context from the pipeline's existing outputs.

Everything the AI layer is allowed to know about a match is derived here, from
the four canonical artifacts a completed run produces:

    analytics.json          match metadata + per-frame positions
    events.json             detected events (passes, recoveries, zone entries, ...)
    player_summary.csv      per-player aggregates (speed, distance, possession)
    possession_summary.csv  team-level possession percentages

The resulting :class:`MatchContext` exposes a compact, human-readable text block
(:meth:`MatchContext.to_prompt_text`) that is injected into Gemini prompts. No
football knowledge is added here - only what the files contain.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.exporters.output_schema import OutputFiles

EVENTS_JSON = "events.json"

_DEFAULT_TEAM_NAMES = {0: "Team A", 1: "Team B", -1: "Unassigned", -2: "Referee"}

_ATTACKING_EVENTS = {"cross", "final_third_entry", "penalty_area_entry", "switch_of_play"}
_DEFENSIVE_EVENTS = {"interception", "recovery"}

_MOMENTUM_BUCKETS = 6


def _safe_read_json(path: str) -> Any | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _safe_read_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None


def load_match_data(game_dir: str) -> dict[str, Any]:
    """Load the raw artifacts from a game folder. Missing files yield ``None``."""
    return {
        "analytics": _safe_read_json(os.path.join(game_dir, OutputFiles.ANALYTICS_JSON)),
        "events": _safe_read_json(os.path.join(game_dir, EVENTS_JSON)),
        "player_summary": _safe_read_csv(os.path.join(game_dir, OutputFiles.PLAYER_SUMMARY)),
        "possession_summary": _safe_read_csv(os.path.join(game_dir, OutputFiles.POSSESSION_SUMMARY)),
    }


@dataclass
class MatchContext:
    game_id: str
    home_team: str = "Team A"
    away_team: str = "Team B"
    available: dict[str, bool] = field(default_factory=dict)
    possession: list[dict[str, Any]] = field(default_factory=list)
    players: list[dict[str, Any]] = field(default_factory=list)
    top_performers: dict[str, Any] = field(default_factory=dict)
    events: dict[str, Any] = field(default_factory=dict)
    momentum: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        return any(self.available.values())

    def team_name(self, team_id: int) -> str:
        if team_id == 0:
            return self.home_team
        if team_id == 1:
            return self.away_team
        return _DEFAULT_TEAM_NAMES.get(team_id, f"Team {team_id}")

    def to_prompt_text(self) -> str:
        return _render_context_text(self)


def _normalize_team_id(raw: Any) -> int | None:
    """Map any team identifier the pipeline emits to a canonical id (0/1/-1/-2)."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    text = str(raw).strip().lower()
    if text in ("home", "team a", "team_0", "team 0", "0"):
        return 0
    if text in ("away", "team b", "team_1", "team 1", "1"):
        return 1
    if text in ("referee", "ref"):
        return -2
    if text.startswith("team "):
        try:
            return int(text.split()[1])
        except (ValueError, IndexError):
            return None
    return None


def _team_names_from_analytics(analytics: dict | None) -> tuple[str, str]:
    home, away = "Team A", "Team B"
    if isinstance(analytics, dict):
        info = analytics.get("match_info") or {}
        h = (info.get("home_team") or "").strip()
        a = (info.get("away_team") or "").strip()
        if h and h.lower() != "home team":
            home = h
        if a and a.lower() != "away team":
            away = a
    return home, away


def _build_possession(df: pd.DataFrame | None, ctx: MatchContext) -> list[dict[str, Any]]:
    if df is None or "team_id" not in df.columns or "possession_pct" not in df.columns:
        return []
    rows = []
    for _, r in df.iterrows():
        tid = int(r["team_id"])
        if tid not in (0, 1):
            continue
        rows.append({
            "team_id": tid,
            "team": ctx.team_name(tid),
            "possession_pct": round(float(r["possession_pct"]), 1),
            "total_frames": int(r.get("total_frames", 0)),
        })
    rows.sort(key=lambda x: x["possession_pct"], reverse=True)
    return rows


def _build_players(df: pd.DataFrame | None, ctx: MatchContext) -> list[dict[str, Any]]:
    if df is None:
        return []
    players = []
    for _, r in df.iterrows():
        role = str(r.get("role", "player")).lower()
        team_id = int(r.get("team_id", -1))
        if role == "referee" or team_id not in (0, 1):
            continue
        players.append({
            "player_id": int(r.get("object_id", -1)),
            "team_id": team_id,
            "team": ctx.team_name(team_id),
            "top_speed_km_h": round(float(r.get("top_speed_km_h", 0.0)), 1),
            "avg_speed_km_h": round(float(r.get("avg_speed_km_h", 0.0)), 1),
            "total_distance_m": round(float(r.get("total_distance_m", 0.0)), 1),
            "poss_pct": round(float(r.get("poss_pct", 0.0)), 1),
            "frames_tracked": int(r.get("total_frames", 0)),
        })
    return players


def _build_top_performers(players: list[dict[str, Any]]) -> dict[str, Any]:
    if not players:
        return {}

    def top(metric: str) -> dict[str, Any] | None:
        ranked = max(players, key=lambda p: p[metric], default=None)
        return ranked if ranked and ranked[metric] > 0 else None

    return {
        "most_distance": top("total_distance_m"),
        "fastest": top("top_speed_km_h"),
        "most_possession": top("poss_pct"),
        "most_tracked": top("frames_tracked"),
    }


def _event_team_id(event: dict) -> int | None:
    for key in ("team", "passer_team", "interceptor_team", "receiver_team"):
        if key in event:
            tid = _normalize_team_id(event.get(key))
            if tid is not None:
                return tid
    return None


def _build_events(events: list | None, ctx: MatchContext) -> dict[str, Any]:
    if not isinstance(events, list) or not events:
        return {}

    by_type: Counter[str] = Counter()
    by_team_type: dict[int, Counter[str]] = defaultdict(Counter)
    timeline: list[dict[str, Any]] = []
    notable_types = _ATTACKING_EVENTS | _DEFENSIVE_EVENTS

    for ev in events:
        if not isinstance(ev, dict):
            continue
        etype = ev.get("type", "unknown")
        by_type[etype] += 1
        tid = _event_team_id(ev)
        if tid in (0, 1):
            by_team_type[tid][etype] += 1
        if etype in notable_types:
            timeline.append({
                "clock": ev.get("game_clock") or _ms_to_clock(ev.get("timestamp_ms")),
                "frame": ev.get("frame"),
                "type": etype,
                "team": ctx.team_name(tid) if tid is not None else "Unknown",
            })

    timeline.sort(key=lambda e: (e["frame"] if e["frame"] is not None else 0))

    team_breakdown = {}
    for tid in (0, 1):
        counts = by_team_type.get(tid)
        if not counts:
            continue
        team_breakdown[ctx.team_name(tid)] = {
            "attacking_actions": sum(counts[t] for t in _ATTACKING_EVENTS),
            "defensive_actions": sum(counts[t] for t in _DEFENSIVE_EVENTS),
            "by_type": dict(counts),
        }

    return {
        "total": sum(by_type.values()),
        "counts_by_type": dict(by_type),
        "by_team": team_breakdown,
        "timeline": timeline,
    }


def _build_momentum(analytics: dict | None, ctx: MatchContext) -> list[dict[str, Any]]:
    """Possession share across time segments, from per-frame in-possession flags."""
    if not isinstance(analytics, dict):
        return []
    frames = analytics.get("frames")
    if not isinstance(frames, list) or len(frames) < _MOMENTUM_BUCKETS:
        return []

    frame_ids = [f.get("frame_id", 0) for f in frames if isinstance(f, dict)]
    if not frame_ids:
        return []
    max_frame = max(frame_ids) or 1
    span = max_frame / _MOMENTUM_BUCKETS

    buckets = [defaultdict(int) for _ in range(_MOMENTUM_BUCKETS)]
    bucket_clock = [None] * _MOMENTUM_BUCKETS

    for f in frames:
        if not isinstance(f, dict):
            continue
        idx = min(int(f.get("frame_id", 0) / span), _MOMENTUM_BUCKETS - 1)
        if bucket_clock[idx] is None:
            bucket_clock[idx] = _ms_to_clock(f.get("timestamp_ms"))
        for p in f.get("players", []):
            if not isinstance(p, dict) or not p.get("in_possession"):
                continue
            tid = _normalize_team_id(p.get("team"))
            if tid in (0, 1):
                buckets[idx][tid] += 1

    momentum = []
    for i, bucket in enumerate(buckets):
        total = bucket.get(0, 0) + bucket.get(1, 0)
        if total == 0:
            continue
        momentum.append({
            "segment": i + 1,
            "from_clock": bucket_clock[i],
            ctx.team_name(0): round(100.0 * bucket.get(0, 0) / total, 1),
            ctx.team_name(1): round(100.0 * bucket.get(1, 0) / total, 1),
        })
    return momentum


def _ms_to_clock(timestamp_ms: Any) -> str | None:
    if timestamp_ms is None:
        return None
    try:
        seconds = int(int(timestamp_ms) / 1000)
    except (TypeError, ValueError):
        return None
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def build_match_context(game_dir: str, game_id: str | None = None) -> MatchContext:
    data = load_match_data(game_dir)
    home, away = _team_names_from_analytics(data["analytics"])

    ctx = MatchContext(
        game_id=game_id or os.path.basename(os.path.normpath(game_dir)),
        home_team=home,
        away_team=away,
        available={
            "analytics": data["analytics"] is not None,
            "events": data["events"] is not None,
            "player_summary": data["player_summary"] is not None,
            "possession_summary": data["possession_summary"] is not None,
        },
    )
    ctx.possession = _build_possession(data["possession_summary"], ctx)
    ctx.players = _build_players(data["player_summary"], ctx)
    ctx.top_performers = _build_top_performers(ctx.players)
    ctx.events = _build_events(data["events"], ctx)
    ctx.momentum = _build_momentum(data["analytics"], ctx)
    return ctx


def _render_context_text(ctx: MatchContext) -> str:
    lines: list[str] = []
    lines.append(f"MATCH: {ctx.home_team} vs {ctx.away_team}")
    lines.append("")

    if ctx.possession:
        lines.append("TEAM POSSESSION:")
        for p in ctx.possession:
            lines.append(f"  - {p['team']}: {p['possession_pct']}% (over {p['total_frames']} frames)")
    else:
        lines.append("TEAM POSSESSION: not available")
    lines.append("")

    if ctx.top_performers:
        lines.append("TOP PERFORMERS:")
        labels = {
            "most_distance": ("Most distance covered", "total_distance_m", "m"),
            "fastest": ("Fastest top speed", "top_speed_km_h", "km/h"),
            "most_possession": ("Highest possession share", "poss_pct", "%"),
            "most_tracked": ("Most frames on screen", "frames_tracked", "frames"),
        }
        for key, (label, metric, unit) in labels.items():
            p = ctx.top_performers.get(key)
            if p:
                lines.append(
                    f"  - {label}: Player {p['player_id']} ({p['team']}) - {p[metric]} {unit}"
                )
    lines.append("")

    if ctx.players:
        lines.append(f"PLAYER STATS ({len(ctx.players)} tracked players):")
        ranked = sorted(ctx.players, key=lambda p: p["total_distance_m"], reverse=True)
        for p in ranked[:14]:
            lines.append(
                f"  - Player {p['player_id']} ({p['team']}): "
                f"distance {p['total_distance_m']} m, "
                f"top speed {p['top_speed_km_h']} km/h, "
                f"avg speed {p['avg_speed_km_h']} km/h, "
                f"possession {p['poss_pct']}%"
            )
    lines.append("")

    if ctx.events:
        lines.append(f"DETECTED EVENTS (total {ctx.events.get('total', 0)}):")
        for etype, count in sorted(ctx.events.get("counts_by_type", {}).items()):
            lines.append(f"  - {etype}: {count}")
        lines.append("")
        if ctx.events.get("by_team"):
            lines.append("EVENTS BY TEAM:")
            for team, stats in ctx.events["by_team"].items():
                lines.append(
                    f"  - {team}: {stats['attacking_actions']} attacking actions, "
                    f"{stats['defensive_actions']} defensive actions"
                )
            lines.append("")
        timeline = ctx.events.get("timeline", [])
        if timeline:
            lines.append("KEY EVENT TIMELINE (chronological):")
            for e in timeline[:40]:
                clock = e["clock"] or f"frame {e['frame']}"
                lines.append(f"  - [{clock}] {e['team']}: {e['type'].replace('_', ' ')}")
            lines.append("")
    else:
        lines.append("DETECTED EVENTS: not available")
        lines.append("")

    if ctx.momentum:
        lines.append("MATCH MOMENTUM (possession share by time segment):")
        for m in ctx.momentum:
            clock = m.get("from_clock") or f"segment {m['segment']}"
            lines.append(
                f"  - from {clock}: {ctx.home_team} {m.get(ctx.home_team, 0)}% / "
                f"{ctx.away_team} {m.get(ctx.away_team, 0)}%"
            )
        lines.append("")

    return "\n".join(lines).strip()
