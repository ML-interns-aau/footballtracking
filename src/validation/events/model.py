"""Common data model shared by both sides of the event-validation harness.

The harness compares two streams of events:

* **ours**       – produced by :class:`src.analytics.events.EventsDetector`
                   and written to ``events.json``.
* **statsbomb**  – ground-truth events from StatsBomb open-data.

Both are normalised into :class:`NormEvent` so the matcher can compare them
without caring about their wildly different native schemas.

Coordinate convention
---------------------
Our pipeline maps the pitch to ``105 m x 68 m`` with the origin at the
top-left (see ``dst_points`` in ``configs/homography.json``).  StatsBomb uses
a ``120 x 80`` grid, also origin top-left.  :func:`statsbomb_to_pitch`
rescales StatsBomb coordinates into our metric pitch so spatial errors are
expressed in metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Canonical event types
# ---------------------------------------------------------------------------
# Types we score in the headline metrics.  These map cleanly and unambiguously
# between our detector and StatsBomb, so precision/recall on them is trustworthy.
SCORED_TYPES: tuple[str, ...] = ("pass", "cross", "interception", "recovery")

# Types we emit that only map to *derived* StatsBomb ground truth (inferred from
# event locations rather than a first-class StatsBomb type).  Reported as
# diagnostics, kept out of the headline numbers to avoid misleading accuracy.
DIAGNOSTIC_TYPES: tuple[str, ...] = (
    "switch_of_play",
    "skill_move",
    "final_third_entry",
    "penalty_area_entry",
)

ALL_TYPES: tuple[str, ...] = SCORED_TYPES + DIAGNOSTIC_TYPES

# Canonical team sides.
HOME = "home"
AWAY = "away"


@dataclass
class NormEvent:
    """A single event normalised to a comparable form.

    Attributes
    ----------
    type:
        Canonical event type, one of :data:`ALL_TYPES`.
    match_time_s:
        Absolute match-clock time in seconds (period-cumulative). For our
        events this is ``clip_start_s + timestamp_ms / 1000``; for StatsBomb
        it is ``minute * 60 + second``.
    team:
        Canonical side (:data:`HOME` / :data:`AWAY`) or ``None`` when unknown.
    x, y:
        Pitch-space position in *our* metric convention (105 x 68, origin
        top-left), or ``None`` if the source has no location for the event.
    source:
        ``"ours"`` or ``"statsbomb"`` – used only for diagnostics.
    raw:
        The original event dict, retained so reports can cite native fields.
    """

    type: str
    match_time_s: float
    team: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    source: str = "ours"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def clock(self) -> str:
        """``MM:SS`` rendering of :attr:`match_time_s` for human-readable reports."""
        total = int(self.match_time_s)
        return f"{total // 60:02d}:{total % 60:02d}"


# StatsBomb pitch dimensions (constant across all open-data matches).
STATSBOMB_PITCH_X = 120.0
STATSBOMB_PITCH_Y = 80.0


def statsbomb_to_pitch(
    x: float,
    y: float,
    pitch_width_m: float = 105.0,
    pitch_height_m: float = 68.0,
) -> tuple[float, float]:
    """Convert a StatsBomb ``[x, y]`` location into our metric pitch space.

    Both systems place the origin at the top-left, so this is a pure rescale.
    """
    return (
        x / STATSBOMB_PITCH_X * pitch_width_m,
        y / STATSBOMB_PITCH_Y * pitch_height_m,
    )
