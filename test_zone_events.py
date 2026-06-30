"""Quick integration test for penalty_area_entry and final_third_entry events."""
from src.analytics.events import EventsDetector


class FakeExporter:
    def __init__(self):
        self.events = []
    def add_event(self, e):
        self.events.append(e)


def test_zone_events():
    ed = EventsDetector(fps=15, pitch_width_m=105, pitch_height_m=68)
    exp = FakeExporter()

    # Establish possession at centre (frames 0-4)
    for f in range(5):
        ed.process_frame(
            frame_idx=f,
            ball_pos_m=(52.0, 34.0),
            player_positions={1: (52.0, 34.0)},
            player_teams={1: "Team 0"},
            ball_speed_kmh=2.0,
            data_exporter=exp,
        )

    assert ed._state == "CONTROLLED", f"Expected CONTROLLED, got {ed._state}"
    assert ed._possessor_id == 1, f"Expected possessor 1, got {ed._possessor_id}"
    print(f"[OK] Possession established: state={ed._state}, possessor={ed._possessor_id}")

    # Move possessor into LEFT penalty area (x=10, y=34)
    exp.events.clear()
    ed.process_frame(
        frame_idx=10,
        ball_pos_m=(10.0, 34.0),
        player_positions={1: (10.0, 34.0)},
        player_teams={1: "Team 0"},
        ball_speed_kmh=2.0,
        data_exporter=exp,
    )

    zone_events = [e for e in exp.events if e["type"] in ("penalty_area_entry", "final_third_entry")]
    pa_events = [e for e in zone_events if e["type"] == "penalty_area_entry"]
    ft_events = [e for e in zone_events if e["type"] == "final_third_entry"]
    
    assert len(pa_events) == 1, f"Expected 1 penalty_area_entry, got {len(pa_events)}"
    assert len(ft_events) == 1, f"Expected 1 final_third_entry, got {len(ft_events)}"
    assert pa_events[0]["player_id"] == 1
    assert pa_events[0]["team"] == "Team 0"
    assert pa_events[0]["entry_xy"] == [10.0, 34.0]
    print(f"[OK] Left PA entry: penalty_area_entry + final_third_entry emitted")

    # Move back to centre - should NOT fire zone events
    exp.events.clear()
    ed.process_frame(
        frame_idx=11,
        ball_pos_m=(52.0, 34.0),
        player_positions={1: (52.0, 34.0)},
        player_teams={1: "Team 0"},
        ball_speed_kmh=2.0,
        data_exporter=exp,
    )
    zone_events2 = [e for e in exp.events if e["type"] in ("penalty_area_entry", "final_third_entry")]
    assert len(zone_events2) == 0, f"Expected 0 zone events at centre, got {len(zone_events2)}"
    print(f"[OK] No zone events at centre (correct)")

    # Move into RIGHT penalty area after cooldown (frame 100)
    exp.events.clear()
    ed.process_frame(
        frame_idx=100,
        ball_pos_m=(95.0, 34.0),
        player_positions={1: (95.0, 34.0)},
        player_teams={1: "Team 0"},
        ball_speed_kmh=2.0,
        data_exporter=exp,
    )
    zone_events3 = [e for e in exp.events if e["type"] in ("penalty_area_entry", "final_third_entry")]
    pa3 = [e for e in zone_events3 if e["type"] == "penalty_area_entry"]
    ft3 = [e for e in zone_events3 if e["type"] == "final_third_entry"]
    assert len(pa3) == 1, f"Expected 1 penalty_area_entry at right PA, got {len(pa3)}"
    assert len(ft3) == 1, f"Expected 1 final_third_entry at right third, got {len(ft3)}"
    assert pa3[0]["entry_xy"] == [95.0, 34.0]
    print(f"[OK] Right PA entry: penalty_area_entry + final_third_entry emitted")

    # Test cooldown: re-enter immediately (frame 101) - should NOT fire
    exp.events.clear()
    # Move out briefly
    ed.process_frame(
        frame_idx=101,
        ball_pos_m=(52.0, 34.0),
        player_positions={1: (52.0, 34.0)},
        player_teams={1: "Team 0"},
        ball_speed_kmh=2.0,
        data_exporter=exp,
    )
    # Re-enter within cooldown
    exp.events.clear()
    ed.process_frame(
        frame_idx=102,
        ball_pos_m=(95.0, 34.0),
        player_positions={1: (95.0, 34.0)},
        player_teams={1: "Team 0"},
        ball_speed_kmh=2.0,
        data_exporter=exp,
    )
    zone_events4 = [e for e in exp.events if e["type"] in ("penalty_area_entry", "final_third_entry")]
    assert len(zone_events4) == 0, f"Expected 0 zone events during cooldown, got {len(zone_events4)}"
    print(f"[OK] Cooldown respected (no re-fire within 50 frames)")

    # Test: player outside y-band should NOT trigger penalty area entry
    exp.events.clear()
    # Move out first
    ed.process_frame(
        frame_idx=200,
        ball_pos_m=(52.0, 34.0),
        player_positions={1: (52.0, 34.0)},
        player_teams={1: "Team 0"},
        ball_speed_kmh=2.0,
        data_exporter=exp,
    )
    exp.events.clear()
    ed.process_frame(
        frame_idx=201,
        ball_pos_m=(5.0, 5.0),  # Inside x-range but outside y-band
        player_positions={1: (5.0, 5.0)},
        player_teams={1: "Team 0"},
        ball_speed_kmh=2.0,
        data_exporter=exp,
    )
    zone_events5 = [e for e in exp.events if e["type"] == "penalty_area_entry"]
    ft5 = [e for e in exp.events if e["type"] == "final_third_entry"]
    assert len(zone_events5) == 0, f"Expected 0 PA events outside y-band, got {len(zone_events5)}"
    assert len(ft5) == 1, f"Expected 1 final_third_entry (y doesn't matter for thirds), got {len(ft5)}"
    print(f"[OK] Penalty area respects y-band constraint; final third ignores y")

    print(f"\n{'='*50}")
    print(f"ALL TESTS PASSED!")
    print(f"{'='*50}")


def test_predicted_ball_guard():
    """A frame whose ball position is predicted (not detected) must not alter
    possession state or emit ball-dependent events."""
    ed = EventsDetector(fps=15)
    exp = FakeExporter()

    # Establish possession on real (detected-ball) frames.
    for f in range(5):
        ed.process_frame(f, (52.0, 34.0), {1: (52.0, 34.0)}, {1: "Team 0"}, 2.0, exp)
    assert ed._state == "CONTROLLED"
    state_before = ed._state
    possessor_before = ed._possessor_id

    # A predicted frame that *looks* like a fast pass to a distant opponent.
    exp.events.clear()
    ed.process_frame(
        6, (80.0, 10.0),
        {1: (52.0, 34.0), 2: (80.0, 10.0)},
        {1: "Team 0", 2: "Team 1"},
        40.0, exp, ball_is_predicted=True,
    )
    assert ed._state == state_before, f"state changed on predicted frame: {ed._state}"
    assert ed._possessor_id == possessor_before
    assert len(exp.events) == 0, f"predicted frame emitted events: {exp.events}"
    print("[OK] predicted-ball frame emits nothing and holds possession state")


if __name__ == "__main__":
    test_zone_events()
    test_predicted_ball_guard()
