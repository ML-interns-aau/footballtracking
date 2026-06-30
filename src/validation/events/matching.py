"""Temporal matching of detected events to ground-truth events, with metrics.

For each canonical event type we pair our events with StatsBomb events that
fall within a time tolerance, minimising total time error (optimal assignment
when SciPy is available, greedy nearest-time otherwise). From the pairing we
derive true positives, false positives (our spurious detections) and false
negatives (missed ground-truth events), then precision / recall / F1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .model import NormEvent, SCORED_TYPES


@dataclass
class MatchedPair:
    ours: NormEvent
    truth: NormEvent
    time_error_s: float
    spatial_error_m: Optional[float]
    team_agrees: Optional[bool]


@dataclass
class TypeResult:
    """Matching outcome and metrics for a single event type."""

    event_type: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched: list[MatchedPair] = field(default_factory=list)
    false_positives: list[NormEvent] = field(default_factory=list)  # ours, unmatched
    false_negatives: list[NormEvent] = field(default_factory=list)  # truth, unmatched

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_time_error_s(self) -> Optional[float]:
        if not self.matched:
            return None
        return sum(m.time_error_s for m in self.matched) / len(self.matched)

    @property
    def mean_spatial_error_m(self) -> Optional[float]:
        errs = [m.spatial_error_m for m in self.matched if m.spatial_error_m is not None]
        return sum(errs) / len(errs) if errs else None

    @property
    def team_agreement_rate(self) -> Optional[float]:
        flags = [m.team_agrees for m in self.matched if m.team_agrees is not None]
        return sum(1 for f in flags if f) / len(flags) if flags else None


def _spatial_error(a: NormEvent, b: NormEvent) -> Optional[float]:
    if None in (a.x, a.y, b.x, b.y):
        return None
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _optimal_pairs(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Return (row, col) assignments minimising total cost.

    Uses SciPy's Hungarian solver when available; otherwise a greedy
    nearest-cost fallback that is good enough for the small event counts here.
    """
    n_rows = len(cost)
    n_cols = len(cost[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        return []
    try:  # optimal assignment
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(np.array(cost))
        return list(zip(rows.tolist(), cols.tolist()))
    except Exception:  # pragma: no cover - greedy fallback
        triples = sorted(
            ((cost[r][c], r, c) for r in range(n_rows) for c in range(n_cols)),
            key=lambda t: t[0],
        )
        used_r: set[int] = set()
        used_c: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for _, r, c in triples:
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))
        return pairs


def match_type(
    ours: list[NormEvent],
    truth: list[NormEvent],
    event_type: str,
    tolerance_s: float,
    match_teams: bool = False,
) -> TypeResult:
    """Match a single event type within ``tolerance_s`` and compute metrics."""
    ours = [e for e in ours if e.type == event_type]
    truth = [e for e in truth if e.type == event_type]

    result = TypeResult(event_type=event_type)
    if not ours and not truth:
        return result

    # Build a cost matrix; pairs outside tolerance (or disagreeing teams when
    # team-aware) are made unselectable via an infinite cost.
    INF = float("inf")
    cost: list[list[float]] = []
    for o in ours:
        row: list[float] = []
        for t in truth:
            dt = abs(o.match_time_s - t.match_time_s)
            feasible = dt <= tolerance_s
            if match_teams and o.team is not None and t.team is not None:
                feasible = feasible and (o.team == t.team)
            row.append(dt if feasible else INF)
        cost.append(row)

    matched_o: set[int] = set()
    matched_t: set[int] = set()
    for r, c in _optimal_pairs(cost) if cost and cost[0] else []:
        if cost[r][c] == INF:
            continue
        o, t = ours[r], truth[c]
        team_agrees = (
            None if (o.team is None or t.team is None) else (o.team == t.team)
        )
        result.matched.append(
            MatchedPair(
                ours=o,
                truth=t,
                time_error_s=abs(o.match_time_s - t.match_time_s),
                spatial_error_m=_spatial_error(o, t),
                team_agrees=team_agrees,
            )
        )
        matched_o.add(r)
        matched_t.add(c)

    result.false_positives = [o for i, o in enumerate(ours) if i not in matched_o]
    result.false_negatives = [t for j, t in enumerate(truth) if j not in matched_t]
    result.tp = len(result.matched)
    result.fp = len(result.false_positives)
    result.fn = len(result.false_negatives)
    return result


@dataclass
class ValidationResult:
    """Per-type results plus a micro-averaged overall summary."""

    run_name: str
    per_type: dict[str, TypeResult]
    tolerance_s: float
    match_teams: bool

    @property
    def overall_tp(self) -> int:
        return sum(r.tp for r in self.per_type.values())

    @property
    def overall_fp(self) -> int:
        return sum(r.fp for r in self.per_type.values())

    @property
    def overall_fn(self) -> int:
        return sum(r.fn for r in self.per_type.values())

    @property
    def overall_precision(self) -> float:
        denom = self.overall_tp + self.overall_fp
        return self.overall_tp / denom if denom else 0.0

    @property
    def overall_recall(self) -> float:
        denom = self.overall_tp + self.overall_fn
        return self.overall_tp / denom if denom else 0.0

    @property
    def overall_f1(self) -> float:
        p, r = self.overall_precision, self.overall_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def validate_events(
    ours: list[NormEvent],
    truth: list[NormEvent],
    run_name: str = "run",
    tolerance_s: float = 3.0,
    match_teams: bool = False,
    types: tuple[str, ...] = SCORED_TYPES,
) -> ValidationResult:
    """Score ``ours`` against ``truth`` across ``types`` and return results."""
    per_type = {
        t: match_type(ours, truth, t, tolerance_s, match_teams) for t in types
    }
    return ValidationResult(
        run_name=run_name,
        per_type=per_type,
        tolerance_s=tolerance_s,
        match_teams=match_teams,
    )
