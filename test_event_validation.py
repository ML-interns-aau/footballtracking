"""Tests for the StatsBomb event-validation harness (src/validation/events).

Self-contained: builds synthetic StatsBomb events and our normalised events,
then asserts the matcher's TP/FP/FN and precision/recall behave as expected.
Run directly (``python test_event_validation.py``) or via pytest.
"""

from src.validation.events.matching import match_type, validate_events
from src.validation.events.model import NormEvent, statsbomb_to_pitch
from src.validation.events.statsbomb import normalize_statsbomb_events
from src.validation.events.ours import normalize_our_events


# --------------------------------------------------------------------------- #
# Coordinate + normalisation
# --------------------------------------------------------------------------- #

def test_coordinate_rescale():
    # StatsBomb centre spot (60, 40) -> our pitch centre (52.5, 34).
    x, y = statsbomb_to_pitch(60.0, 40.0)
    assert abs(x - 52.5) < 1e-6 and abs(y - 34.0) < 1e-6
    # Corners map to corners.
    assert statsbomb_to_pitch(0, 0) == (0.0, 0.0)
    assert abs(statsbomb_to_pitch(120, 80)[0] - 105.0) < 1e-6
    print("[OK] coordinate rescale")


def test_statsbomb_type_mapping():
    raw = [
        {"type": {"name": "Pass"}, "minute": 0, "second": 10,
         "team": {"name": "A"}, "location": [60, 40], "pass": {}},
        {"type": {"name": "Pass"}, "minute": 0, "second": 12,
         "team": {"name": "A"}, "location": [110, 40], "pass": {"cross": True}},
        {"type": {"name": "Pass"}, "minute": 0, "second": 14,
         "team": {"name": "A"}, "pass": {"outcome": {"name": "Incomplete"}}},  # dropped
        {"type": {"name": "Interception"}, "minute": 0, "second": 16, "team": {"name": "B"}},
        {"type": {"name": "Ball Recovery"}, "minute": 0, "second": 18, "team": {"name": "B"}},
        {"type": {"name": "Pressure"}, "minute": 0, "second": 20, "team": {"name": "B"}},  # dropped
    ]
    norm = normalize_statsbomb_events(raw, team_map={"A": "home", "B": "away"})
    types = sorted(e.type for e in norm)
    assert types == ["cross", "interception", "pass", "recovery"], types
    # completed pass keeps team side and rescaled location
    p = next(e for e in norm if e.type == "pass")
    assert p.team == "home" and abs(p.x - 52.5) < 1e-6
    print("[OK] statsbomb type mapping + outcome filtering")


def test_switch_emits_pass_and_switch():
    raw = [{"type": {"name": "Pass"}, "minute": 1, "second": 0,
            "team": {"name": "A"}, "location": [30, 10],
            "pass": {"switch": True}}]
    norm = normalize_statsbomb_events(raw, team_map={"A": "home"})
    assert sorted(e.type for e in norm) == ["pass", "switch_of_play"]
    print("[OK] switch emits both pass and switch_of_play")


def test_our_events_clock_shift():
    raw = [
        {"type": "pass", "timestamp_ms": 2000, "passer_team": "Team 0",
         "start_xy": [50.0, 30.0]},
        {"type": "cross", "timestamp_ms": 5000, "team": "Team 1",
         "origin_x_m": 90.0, "origin_y_m": 60.0},
    ]
    norm = normalize_our_events(raw, clip_start_s=600.0)  # clip starts at 10:00
    assert norm[0].match_time_s == 602.0 and norm[0].team == "home"
    assert norm[1].match_time_s == 605.0 and norm[1].team == "away"
    assert norm[1].x == 90.0 and norm[1].y == 60.0
    print("[OK] our events clock shift + team/xy extraction")


# --------------------------------------------------------------------------- #
# Matching + metrics
# --------------------------------------------------------------------------- #

def _ev(t, ts, src="ours", team=None, xy=(None, None)):
    return NormEvent(type=t, match_time_s=ts, team=team, x=xy[0], y=xy[1], source=src)


def test_match_perfect():
    ours = [_ev("pass", 10.0), _ev("pass", 20.0)]
    truth = [_ev("pass", 10.4, "statsbomb"), _ev("pass", 19.8, "statsbomb")]
    r = match_type(ours, truth, "pass", tolerance_s=3.0)
    assert (r.tp, r.fp, r.fn) == (2, 0, 0)
    assert r.precision == 1.0 and r.recall == 1.0 and r.f1 == 1.0
    print("[OK] perfect match")


def test_match_fp_and_fn():
    # one true positive, one spurious detection (FP), one missed truth (FN)
    ours = [_ev("pass", 10.0), _ev("pass", 50.0)]            # 50.0 is spurious
    truth = [_ev("pass", 10.5, "statsbomb"), _ev("pass", 80.0, "statsbomb")]  # 80.0 missed
    r = match_type(ours, truth, "pass", tolerance_s=3.0)
    assert (r.tp, r.fp, r.fn) == (1, 1, 1), (r.tp, r.fp, r.fn)
    assert r.precision == 0.5 and r.recall == 0.5
    assert r.false_positives[0].match_time_s == 50.0
    assert r.false_negatives[0].match_time_s == 80.0
    print("[OK] false positive + false negative accounting")


def test_match_optimal_assignment():
    # Two ours near two truths; greedy-by-pair could mispair, optimal shouldn't.
    ours = [_ev("pass", 10.0), _ev("pass", 11.0)]
    truth = [_ev("pass", 11.2, "statsbomb"), _ev("pass", 10.1, "statsbomb")]
    r = match_type(ours, truth, "pass", tolerance_s=3.0)
    assert (r.tp, r.fp, r.fn) == (2, 0, 0)
    assert r.mean_time_error_s is not None and r.mean_time_error_s < 0.25
    print("[OK] optimal (low-error) assignment")


def test_team_aware_matching():
    ours = [_ev("pass", 10.0, team="home")]
    truth = [_ev("pass", 10.2, "statsbomb", team="away")]
    # team-blind: matches
    assert match_type(ours, truth, "pass", 3.0, match_teams=False).tp == 1
    # team-aware: rejected -> FP + FN
    r = match_type(ours, truth, "pass", 3.0, match_teams=True)
    assert (r.tp, r.fp, r.fn) == (0, 1, 1)
    print("[OK] team-aware matching gate")


def test_validate_events_overall():
    ours = [_ev("pass", 10.0), _ev("cross", 30.0), _ev("recovery", 40.0)]
    truth = [_ev("pass", 10.3, "statsbomb"), _ev("cross", 30.2, "statsbomb"),
             _ev("interception", 35.0, "statsbomb")]
    res = validate_events(ours, truth, run_name="t", tolerance_s=3.0)
    assert res.per_type["pass"].tp == 1
    assert res.per_type["cross"].tp == 1
    assert res.per_type["recovery"].fp == 1   # our recovery has no truth
    assert res.per_type["interception"].fn == 1  # truth interception missed
    assert res.overall_tp == 2 and res.overall_fp == 1 and res.overall_fn == 1
    print(f"[OK] overall metrics: P={res.overall_precision:.2f} R={res.overall_recall:.2f}")


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{'='*50}\nALL {len(tests)} VALIDATION TESTS PASSED!\n{'='*50}")


if __name__ == "__main__":
    run_all()
