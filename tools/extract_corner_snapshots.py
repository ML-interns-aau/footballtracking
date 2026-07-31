"""Extract two key-frame snapshots from a corner-kick clip: the moment the
ball is struck (kick frame) and the frame of first contact by another player
afterwards (contact frame).

All timing is computed on a camera-motion-compensated, decoy-filtered ball
trajectory (see src/engine/camera_motion.py, src/engine/detector.py's pitch/
LED gates, src/engine/ball_tracker.py's clusterer+smoother, and
src/engine/event_detector.py) rather than raw pixel motion, which is
corrupted by camera pan and by generic-detector decoys (sideline spare balls,
sponsor logos on perimeter LED boards).

Usage:
    python tools/extract_corner_snapshots.py --input clip_1.mp4 \
        --output_dir results/corner_snapshots --model models/yolo11l.pt

    # Override an auto-detected frame if it looks wrong on ball_speed_debug.png:
    python tools/extract_corner_snapshots.py --input clip_1.mp4 \
        --kick_frame 42 --contact_frame 57
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.detector import FootballDetector
from src.engine.tracker import FootballTracker
from src.engine.camera_motion import CameraMotionEstimator
from src.engine.ball_tracker import BallCandidateClusterer, CompensatedBallSmoother
from src.engine.event_detector import EventTimingDetector, compensated_speed_series



def _get_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"



def run_pass_one(video_path: Path, args, device: str):
    detector = FootballDetector(
        model_path=args.model, conf=args.conf, iou=args.iou, device=device,
        imgsz=args.imgsz,
        use_pitch_gate=args.enable_pitch_gate,
        use_led_gate=args.enable_led_gate,
        ball_weights=args.ball_weights,
        gate_offpitch_players=args.filter_offpitch_players,
        ball_conf=args.ball_conf,
    )
    tracker = FootballTracker()
    cam_estimator = CameraMotionEstimator(motion_model=args.motion_model)
    clusterer = BallCandidateClusterer(
        cluster_radius_px=args.cluster_radius_px,
        early_frac=args.decoy_early_frac,
        late_frac=args.decoy_late_frac,
        min_frames_for_decoy=args.decoy_min_frames,
        min_span_frac_for_decoy=args.decoy_min_span_frac,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    per_frame_players: list[dict[int, tuple[float, float]]] = []
    per_frame_player_boxes: list[dict[int, tuple[float, float, float, float]]] = []
    per_frame_raw_players: list[list[tuple[float, float, float, float, float]]] = []
    per_frame_ball_candidates: list[list[dict]] = []
    cam_motion_series: list[tuple[float, float]] = []
    cum_offset_series: list[tuple[float, float]] = []
    n_frames = 0

    prev_gray = None
    cum_dx = cum_dy = 0.0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(frame)
        tracked = tracker.update(detections)

        player_boxes = []
        if detections.class_id is not None:
            for i in np.where(detections.class_id == 0)[0]:
                player_boxes.append(tuple(detections.xyxy[i]))

        if prev_gray is None:
            dx, dy = 0.0, 0.0
        else:
            dx, dy, _ = cam_estimator.estimate(prev_gray, gray, player_boxes=player_boxes)
        cum_dx += dx
        cum_dy += dy
        cam_motion_series.append((dx, dy))
        cum_offset_series.append((cum_dx, cum_dy))

        candidates = []
        above_pitch_flags = detections.data.get("above_pitch") if detections.data is not None else None
        if detections.class_id is not None:
            for i in np.where(detections.class_id == 32)[0]:
                x1, y1, x2, y2 = detections.xyxy[i]
                rx, ry = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                conf = float(detections.confidence[i]) if detections.confidence is not None else 1.0
                comp_xy = (rx - cum_dx, ry - cum_dy)
                above_pitch = bool(above_pitch_flags[i]) if above_pitch_flags is not None else False
                cid = clusterer.add(idx, comp_xy, conf, above_pitch=above_pitch)
                candidates.append({
                    "raw_xy": (rx, ry), "comp_xy": comp_xy, "confidence": conf, "cluster_id": cid,
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                })

        if args.tile_ball_detection:
            tiled = detector.detect_ball_tiled(
                frame, grid_rows=args.tile_grid_rows, grid_cols=args.tile_grid_cols,
                overlap_frac=args.tile_overlap_frac,
            )
            existing_centers = [c["raw_xy"] for c in candidates]
            for i in range(len(tiled)):
                x1, y1, x2, y2 = tiled.xyxy[i]
                rx, ry = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if any(np.hypot(rx - ex, ry - ey) < args.cluster_radius_px for ex, ey in existing_centers):
                    continue
                conf = float(tiled.confidence[i]) if tiled.confidence is not None else 1.0
                comp_xy = (rx - cum_dx, ry - cum_dy)
                cid = clusterer.add(idx, comp_xy, conf)
                candidates.append({
                    "raw_xy": (rx, ry), "comp_xy": comp_xy, "confidence": conf, "cluster_id": cid,
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                })

        per_frame_ball_candidates.append(candidates)

        raw_players: list[tuple[float, float, float, float, float]] = []
        if detections.class_id is not None:
            for i in np.where(detections.class_id == 0)[0]:
                x1, y1, x2, y2 = detections.xyxy[i]
                conf = float(detections.confidence[i]) if detections.confidence is not None else 1.0
                raw_players.append((float(x1), float(y1), float(x2), float(y2), conf))
        per_frame_raw_players.append(raw_players)

        players: dict[int, tuple[float, float]] = {}
        player_boxes: dict[int, tuple[float, float, float, float]] = {}
        for i in range(len(tracked)):
            class_id = int(tracked.class_id[i])  if tracked.class_id  is not None else -1
            tid      = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None
            if class_id != 0 or tid is None:
                continue
            x1, y1, x2, y2 = tracked.xyxy[i]
            players[tid] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            player_boxes[tid] = (float(x1), float(y1), float(x2), float(y2))
        per_frame_players.append(players)
        per_frame_player_boxes.append(player_boxes)

        prev_gray = gray
        idx += 1

    n_frames = idx
    cap.release()
    return {
        "video_path": video_path,
        "fps": fps,
        "frame_size": (width, height),
        "n_frames": n_frames,
        "players": per_frame_players,
        "player_boxes": per_frame_player_boxes,
        "raw_players": per_frame_raw_players,
        "ball_candidates": per_frame_ball_candidates,
        "cam_motion_series": cam_motion_series,
        "cum_offset_series": cum_offset_series,
        "clusterer": clusterer,
        "gate_stats": dict(detector.last_gate_stats),
    }



def run_pass_two(pass1: dict, args) -> list[dict]:
    cluster_summary = pass1["clusterer"].classify()
    decoy_ids = {cid for cid, info in cluster_summary.items() if info["is_decoy"]}

    cluster_points: dict[int, list[tuple[int, tuple[float, float]]]] = {}
    for f_idx, cands in enumerate(pass1["ball_candidates"]):
        for c in cands:
            cluster_points.setdefault(c["cluster_id"], []).append((f_idx, c["comp_xy"]))
    for pts in cluster_points.values():
        pts.sort(key=lambda p: p[0])

    def local_max_speed(cluster_id: int, center_idx: int, radius: int) -> float:
        pts = cluster_points.get(cluster_id, [])
        frame_idxs = [p[0] for p in pts]
        lo = bisect.bisect_left(frame_idxs, center_idx - radius)
        hi = bisect.bisect_right(frame_idxs, center_idx + radius)
        window = pts[lo:hi]
        if len(window) < 2:
            return 0.0
        best = 0.0
        for i in range(1, len(window)):
            f0, p0 = window[i - 1]
            f1, p1 = window[i]
            gap = max(1, f1 - f0)
            best = max(best, float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) / gap)
        return best

    smoother = CompensatedBallSmoother(max_missed=args.max_missed, diverge_px=args.diverge_px)
    n = pass1["n_frames"]

    cap = cv2.VideoCapture(str(pass1["video_path"]))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot re-open video: {pass1['video_path']}")

    frames: list[dict] = []
    active_cluster_id = None
    active_last_seen = -10**9
    for idx in range(n):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Video ended early during pass 2 at frame {idx} (expected {n} frames)")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        candidates = [c for c in pass1["ball_candidates"][idx] if c["cluster_id"] not in decoy_ids]
        comp_measurement = raw_measurement = None
        if candidates:
            in_continuity = active_cluster_id is not None and idx - active_last_seen <= args.candidate_continuity_ttl
            same_cluster = [c for c in candidates if c["cluster_id"] == active_cluster_id] if in_continuity else []
            if same_cluster:
                best = max(same_cluster, key=lambda c: c["confidence"])
            else:
                eligible = [
                    c for c in candidates
                    if not in_continuity or local_max_speed(c["cluster_id"], idx, radius=5) >= args.min_flight_speed_px
                ]
                best = max(eligible, key=lambda c: c["confidence"]) if eligible else None
            if best is not None:
                comp_measurement = best["comp_xy"]
                raw_measurement = best["raw_xy"]
                active_cluster_id = best["cluster_id"]
                active_last_seen = idx

        cum_offset = pass1["cum_offset_series"][idx]
        comp_x, comp_y, is_predicted = smoother.update(gray, comp_measurement, raw_measurement, cum_offset)
        raw_xy = (comp_x + cum_offset[0], comp_y + cum_offset[1])

        frames.append({
            "idx": idx,
            "raw_xy": raw_xy,
            "comp_xy": (comp_x, comp_y),
            "is_predicted": is_predicted,
            "players": pass1["players"][idx],
            "player_boxes": pass1["player_boxes"][idx],
            "raw_players": pass1["raw_players"][idx],
        })

    cap.release()
    return frames, cluster_summary, decoy_ids



_CACHE_VERSION = 9

_PASS12_ARG_NAMES = [
    "model", "conf", "iou", "imgsz", "device", "motion_model", "ball_weights",
    "enable_pitch_gate", "enable_led_gate", "cluster_radius_px",
    "candidate_continuity_ttl", "decoy_early_frac", "decoy_late_frac",
    "decoy_min_frames", "decoy_min_span_frac", "max_missed", "diverge_px",
    "tile_ball_detection", "tile_grid_rows", "tile_grid_cols", "tile_overlap_frac",
    "filter_offpitch_players", "ball_conf",
]


def _pass12_cache_path(video_path: Path, args: argparse.Namespace) -> Path:
    key_parts = {name: getattr(args, name) for name in _PASS12_ARG_NAMES}
    key_parts["input"] = str(video_path.resolve())
    key_parts["input_mtime"] = video_path.stat().st_mtime
    key_parts["cache_version"] = _CACHE_VERSION
    digest = hashlib.sha1(json.dumps(key_parts, sort_keys=True, default=str).encode()).hexdigest()[:16]
    cache_dir = Path(".cache") / "pass12"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{video_path.stem}_{digest}.pkl"


def get_pass12(video_path: Path, args: argparse.Namespace, device: str):
    """Returns (frames, cluster_summary, decoy_ids, gate_stats, fps, frame_size, n_frames,
    cam_motion_series, ball_candidates), from cache when the pass1/2-affecting args are
    unchanged, else computes and caches it."""
    cache_path = _pass12_cache_path(video_path, args)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        print(f"[INFO] Loaded cached pass1+2 result from {cache_path} (skipping detection).")
        return cached

    pass1 = run_pass_one(video_path, args, device)
    n = pass1["n_frames"]
    if n == 0:
        raise ValueError("No frames read from video.")
    frames, cluster_summary, decoy_ids = run_pass_two(pass1, args)
    result = (
        frames, cluster_summary, decoy_ids, pass1["gate_stats"], pass1["fps"],
        pass1["frame_size"], n, pass1["cam_motion_series"], pass1["ball_candidates"],
    )
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[INFO] Cached pass1+2 result to {cache_path}.")
    return result



def find_kicker_id(frames: list[dict], kick_frame: int, kicker_window: int) -> int | None:
    lo = max(0, kick_frame - kicker_window)
    window = frames[lo:kick_frame + 1]
    if not window:
        return None

    tally: dict[int, list[float]] = {}
    for f in window:
        bx, by = f["raw_xy"]
        for tid, (px, py) in f["players"].items():
            tally.setdefault(tid, []).append(float(np.hypot(px - bx, py - by)))

    best_tid, best_mean = None, None
    for tid, dists in tally.items():
        if len(dists) < max(1, len(window) // 2):
            continue
        mean_d = float(np.mean(dists))
        if best_mean is None or mean_d < best_mean:
            best_mean, best_tid = mean_d, tid
    return best_tid


def find_kicker(
    frames: list[dict], kick_frame: int, kicker_window: int,
    max_dist_px: float = 150.0, tracked_match_px: float = 30.0,
) -> dict:
    """Finds the actual corner-taker near the ball at/just before the kick,
    using raw (pre-tracking, but pitch-gated) person detections rather than
    the ByteTrack-confirmed "players" set find_kicker_id relies on.

    This matters because the corner-taker can be small, distant, or right at
    the extreme edge of frame -- exactly the conditions under which YOLO's
    confidence is chronically low (~0.15-0.24 here), which keeps the person
    detected every few frames but almost always below ByteTrack's track
    activation threshold, so they never hold a stable id and silently vanish
    from "players" entirely, even though "who was standing at the corner"
    doesn't actually require a multi-frame identity -- just a good detection
    at or near the kick.

    Returns {"box", "tracker_id", "frame"}: box is the raw detection for
    drawing; tracker_id is filled in only if some ByteTrack-confirmed player
    happens to coincide with that position (kept so the existing contact-
    detection exclusion logic -- which excludes the kicker from the tracked
    "players" search -- still has something to match against; None is a safe
    no-op there since an untracked kicker was never going to appear in that
    search anyway).
    """
    for offset in range(0, kicker_window + 1):
        f_idx = kick_frame - offset
        if f_idx < 0:
            break
        f = frames[f_idx]
        raw = f["raw_players"]
        if not raw:
            continue
        bx, by = f["raw_xy"]
        best = min(raw, key=lambda p: np.hypot((p[0] + p[2]) / 2 - bx, (p[1] + p[3]) / 2 - by))
        cx, cy = (best[0] + best[2]) / 2, (best[1] + best[3]) / 2
        d = np.hypot(cx - bx, cy - by)
        if d <= max_dist_px:
            tracker_id = None
            for tid, (px, py) in f["players"].items():
                if np.hypot(px - cx, py - cy) <= tracked_match_px:
                    tracker_id = tid
                    break
            return {"box": best[:4], "tracker_id": tracker_id, "frame": f_idx}
    return {"box": None, "tracker_id": None, "frame": None}


def _nearest_player(frame: dict, exclude_id: int | None = None) -> tuple[int | None, float | None]:
    bx, by = frame["raw_xy"]
    best_tid, best_dist = None, None
    for tid, (px, py) in frame["players"].items():
        if tid == exclude_id:
            continue
        d = float(np.hypot(px - bx, py - by))
        if best_dist is None or d < best_dist:
            best_dist, best_tid = d, tid
    return best_tid, best_dist



def find_ball_detection_at(
    frame_data: dict,
    ball_candidates_for_frame: list[dict],
    decoy_ids: set[int],
) -> tuple[tuple[float, float, float, float] | None, float | None]:
    """Returns (bbox, confidence) for the real ball detection backing this
    frame's smoothed position, or (None, None) if the frame's ball position
    is Kalman/optical-flow predicted rather than a genuine detection."""
    if frame_data["is_predicted"]:
        return None, None
    real = [c for c in ball_candidates_for_frame if c["cluster_id"] not in decoy_ids]
    if not real:
        return None, None
    rx, ry = frame_data["raw_xy"]
    best = min(real, key=lambda c: np.hypot(c["raw_xy"][0] - rx, c["raw_xy"][1] - ry))
    return best["bbox"], best["confidence"]


def _draw_tag(frame: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_ball_glow(frame: np.ndarray, center: tuple[float, float], radius: int, color: tuple[int, int, int]) -> None:
    """Bright glow ring around the ball -- a soft translucent halo (two
    blended circles) plus a crisp solid ring, rather than a plain bounding box."""
    cx, cy = int(center[0]), int(center[1])
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius * 3, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), radius * 2, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.circle(frame, (cx, cy), radius, color, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)


def _draw_event_marker(frame: np.ndarray, anchor: tuple[int, int], label: str, color: tuple[int, int, int]) -> None:
    """Small downward-triangle marker + role label above a player -- no box,
    no tracker id, just enough to point at who's involved in the event."""
    ax, ay = anchor
    cv2.drawMarker(frame, (ax, ay), color, markerType=cv2.MARKER_TRIANGLE_DOWN, markerSize=16, thickness=2,
                    line_type=cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    h, w = frame.shape[:2]
    label_x = int(np.clip(ax - tw // 2, 2, max(2, w - tw - 2)))
    _draw_tag(frame, label, (label_x, max(ay - 14, 12)), color)


def draw_annotated_snapshot(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    event_label: str,
    player_boxes: dict[int, tuple[float, float, float, float]],
    ball_xy: tuple[float, float],
    ball_bbox: tuple[float, float, float, float] | None,
    ball_confidence: float | None,
    highlight_id: int | None,
    highlight_role: str,
    include_label: bool = True,
    extra_highlights: dict[int, tuple[str, tuple[int, int, int]]] | None = None,
    manual_markers: list[tuple[tuple[float, float, float, float], str, tuple[int, int, int]]] | None = None,
) -> np.ndarray:
    """Clean football-analytics overlay: a bright red glow + confidence on the
    ball, and a small marker + role label above the kicker/contact player.
    No player boxes or tracker ids are drawn for anyone.

    extra_highlights lets more than one tracker id be marked at once (e.g.
    kicker AND contact player simultaneously across a whole-clip frame dump),
    each with its own role label and BGR color -- looked up in player_boxes.

    manual_markers draws a marker at a fixed box directly, bypassing
    player_boxes/tracker ids entirely -- for a player (e.g. the corner-taker)
    who was identified from a raw detection rather than a stable track.
    """
    annotated = frame.copy()
    BALL_COLOR = (0, 0, 255)
    KICKER_COLOR = (0, 215, 255)
    CONTACT_COLOR = (255, 0, 255)

    highlights: dict[int, tuple[str, tuple[int, int, int]]] = dict(extra_highlights or {})
    if highlight_id is not None:
        color = CONTACT_COLOR if highlight_role.upper() == "CONTACT" else KICKER_COLOR
        highlights[highlight_id] = (highlight_role, color)

    for tid, (role_label, color) in highlights.items():
        box = player_boxes.get(tid)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        anchor = (int((x1 + x2) / 2), max(int(y1) - 8, 12))
        _draw_event_marker(annotated, anchor, role_label, color)

    for box, role_label, color in (manual_markers or []):
        x1, y1, x2, y2 = box
        anchor = (int((x1 + x2) / 2), max(int(y1) - 8, 12))
        _draw_event_marker(annotated, anchor, role_label, color)

    if ball_bbox is not None:
        bx1, by1, bx2, by2 = ball_bbox
        radius = max(6, int(max(bx2 - bx1, by2 - by1) / 2) + 3)
    else:
        radius = 8
    _draw_ball_glow(annotated, ball_xy, radius, BALL_COLOR)
    bcx, bcy = int(ball_xy[0]), int(ball_xy[1])
    conf_text = f"ball {ball_confidence:.2f}" if ball_confidence is not None else "ball (predicted, no detection)"
    _draw_tag(annotated, conf_text, (bcx + radius + 6, max(bcy - radius, 12)), BALL_COLOR)

    if include_label:
        header = f"{event_label}  |  frame {frame_idx}  |  t={frame_idx / fps:.2f}s"
        annotated = draw_label(annotated, header)
    return annotated


def draw_label(frame: np.ndarray, text: str) -> np.ndarray:
    frame = frame.copy()
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 12
    x, y = pad, pad + text_h
    cv2.rectangle(frame, (0, 0), (x + text_w + pad, y + baseline + pad), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def save_debug_plot(
    frames: list[dict],
    cam_motion_series: list[tuple[float, float]],
    kick_frame: int, contact_frame: int | None,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(frames)
    x = np.arange(n)
    real_mask = np.array([not f["is_predicted"] for f in frames])

    raw_positions = [f["raw_xy"] for f in frames]
    comp_positions = [f["comp_xy"] for f in frames]
    v_pixel = np.zeros(n)
    for i in range(1, n):
        x0, y0 = raw_positions[i - 1]
        x1, y1 = raw_positions[i]
        v_pixel[i] = np.hypot(x1 - x0, y1 - y0)
    v_comp = compensated_speed_series(comp_positions)
    cam_pan = np.array([np.hypot(dx, dy) for dx, dy in cam_motion_series])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, v_pixel, color="#999999", linewidth=1.0, label="raw pixel speed", alpha=0.7)
    ax.plot(x, cam_pan, color="#d62728", linewidth=1.2, label="camera pan speed")
    ax.plot(x, v_comp, color="#3366cc", linewidth=1.8, label="compensated ball speed")
    ax.scatter(x[~real_mask], v_comp[~real_mask], s=10, color="#d62728", marker="x",
               label="predicted (Kalman/optical-flow)", zorder=3)

    ax.axvline(kick_frame, color="black", linestyle="--", linewidth=1.5, label=f"kick frame ({kick_frame})")
    if contact_frame is not None:
        ax.axvline(contact_frame, color="purple", linestyle="--", linewidth=1.5, label=f"contact frame ({contact_frame})")

    ax.set_xlabel("frame index")
    ax.set_ylabel("speed (px/frame)")
    ax.set_title("Camera-compensated ball speed vs. frame")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract kick-moment and first-contact snapshots from a corner-kick clip.")
    p.add_argument("--input",        required=True, help="Path to the corner-kick video clip.")
    p.add_argument("--output_dir",   default="results/corner_snapshots")
    p.add_argument("--model",        default="models/yolo11l.pt")
    p.add_argument("--conf",         type=float, default=0.15)
    p.add_argument("--ball-conf",    type=float, default=None,
                    help="Confidence floor for the ball model specifically (both full-frame and tiled). "
                         "Defaults to --conf if not given.")
    p.add_argument("--iou",          type=float, default=0.40)
    p.add_argument("--imgsz",        type=int, default=1280)
    p.add_argument("--device",       default="auto", help="auto|cpu|cuda")

    p.add_argument("--motion_model", choices=["affine", "homography"], default="affine")
    p.add_argument("--ball_weights", default=None, help="Optional football/SoccerNet-finetuned ball checkpoint.")
    p.add_argument("--enable-pitch-gate", action="store_true")
    p.add_argument("--enable-led-gate",   action="store_true")

    p.add_argument("--filter-offpitch-players", action=argparse.BooleanOptionalAction, default=True,
                    help="Reject person detections standing outside the pitch surface (bench, sideline "
                         "staff, subs) so they don't pollute tracking, kicker/contact-player lookup, or "
                         "annotated snapshots. Use --no-filter-offpitch-players to disable.")

    p.add_argument("--tile-ball-detection", action="store_true",
                    help="Also run ball detection on overlapping frame tiles to recover a small/fast ball "
                         "a full-frame pass misses; merged in as extra candidates for the same clusterer.")
    p.add_argument("--tile-grid-rows", type=int, default=2)
    p.add_argument("--tile-grid-cols", type=int, default=3)
    p.add_argument("--tile-overlap-frac", type=float, default=0.2,
                    help="Overlap between adjacent tiles as a fraction of tile size, so a ball straddling "
                         "a tile boundary isn't cut in half in every tile that contains it.")

    p.add_argument("--kick_frame",    type=int, default=None, help="Manual override; skips kick-frame detection.")
    p.add_argument("--contact_frame", type=int, default=None, help="Manual override; skips contact-frame detection.")

    p.add_argument("--baseline-window",   type=int,   default=40)
    p.add_argument("--stationary-thresh", type=float, default=4.0)
    p.add_argument("--kick-sigma",        type=float, default=4.0)
    p.add_argument("--persistence-frames", type=int,  default=3)
    p.add_argument("--max-gap-frames",    type=int,   default=5)
    p.add_argument("--corner-margin-frac", type=float, default=0.22)

    p.add_argument("--kicker-window",        type=int,   default=10)
    p.add_argument("--ballistic-fit-frames", type=int,   default=8)
    p.add_argument("--ballistic-fit-window-frames", type=int, default=20,
                    help="How many frames past the kick the fit-gathering walk may range over, "
                         "bounded separately from --contact-search-frames so a long detection gap "
                         "can't make the fit consume real points near/after the actual contact.")
    p.add_argument("--residual-thresh-px",   type=float, default=15.0)
    p.add_argument("--contact-proximity-px", type=float, default=60.0)
    p.add_argument("--contact-search-frames", type=int,  default=60)
    p.add_argument("--contact-persistence-frames", type=int, default=3,
                    help="Consecutive frames the residual departure must hold before it's accepted as contact.")
    p.add_argument("--reversal-angle-thresh-deg", type=float, default=80.0,
                    help="Angle (degrees) between consecutive real-to-real ball velocity vectors "
                         "that counts as a sharp redirect (the primary contact signal).")
    p.add_argument("--min-flight-speed-px",  type=float, default=6.0,
                    help="Velocity vectors slower than this are too noisy to judge a reversal angle from and are skipped.")
    p.add_argument("--max-segment-gap-frames", type=int, default=3,
                    help="Reversal check only compares real-to-real segments both spanning at most this many "
                         "frames, so an average velocity across a longer detection gap isn't compared against "
                         "an adjacent short one.")
    p.add_argument("--arc-break-residual-px", type=float, default=20.0,
                    help="How far (px) a real detection must depart the RANSAC-fitted ballistic arc, beyond "
                         "the arc's own inlier threshold, to count as a contact-candidate break.")
    p.add_argument("--contact-max-gap-frames", type=int, default=20,
                    help="Max frames between the on-arc point and the post-break point for a contact "
                         "candidate -- a real touch can hide the ball for a few frames (occluded by the body "
                         "making contact), so this is deliberately looser than --max-segment-gap-frames.")
    p.add_argument("--contact-min-score", type=float, default=0.35,
                    help="Minimum combined score (arc-break + proximity + gap-bracketing + motion-spike) to "
                         "accept a contact candidate; below this, contact_frame is reported as null rather "
                         "than guessing.")
    p.add_argument("--motion-spike-weight", type=float, default=0.15,
                    help="Weight of the player-own-motion-spike corroborating signal in the contact score.")

    p.add_argument("--cluster-radius-px",  type=float, default=60.0)
    p.add_argument("--candidate-continuity-ttl", type=int, default=15,
                    help="Frames to keep preferring the previously-active ball cluster before re-latching freely.")
    p.add_argument("--decoy-early-frac",   type=float, default=0.10)
    p.add_argument("--decoy-late-frac",    type=float, default=0.60)
    p.add_argument("--decoy-min-frames",   type=int,   default=3)
    p.add_argument("--decoy-min-span-frac", type=float, default=0.40,
                    help="A static cluster spanning this fraction of the clip (regardless of when it starts) is a decoy.")

    p.add_argument("--max-missed",  type=int,   default=30)
    p.add_argument("--diverge-px",  type=float, default=250.0)

    p.add_argument("--no-label", action="store_true",
                    help="Don't burn frame#/time-stamp labels into the saved snapshot images.")
    p.add_argument("--dump-all-frames", action="store_true",
                    help="Also save every frame of the clip, annotated the same way as the kick/contact "
                         "snapshots (players + ball + confidence, kicker/contact player highlighted "
                         "throughout), into <output_dir>/all_frames/ -- for manually scrubbing the whole "
                         "clip to sanity-check the auto-picked frames.")
    p.add_argument("--contact-window-frames", type=int, default=15,
                    help="Also save this many annotated frames before and after the contact frame (if one "
                         "was determined) into <output_dir>/contact_window/, for closely reviewing exactly "
                         "what happens around the picked moment.")
    p.add_argument("--kick-window-frames", type=int, default=0,
                    help="Also save this many annotated frames before and after the kick frame into "
                         "<output_dir>/kick_window/, for manually confirming the auto-picked kick frame the "
                         "same way --contact-window-frames lets you confirm contact.")

    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    """Runs the full pipeline and returns a result dict (no file I/O beyond
    what's needed internally). Used by both main() and tools/validate.py."""
    video_path = Path(args.input)
    if not video_path.exists():
        raise FileNotFoundError(f"Input not found: {video_path}")

    device = _get_device(args.device)
    frames, cluster_summary, decoy_ids, gate_stats, fps, frame_size, n, cam_motion_series, ball_candidates = \
        get_pass12(video_path, args, device)

    any_ball_detected = any(not f["is_predicted"] for f in frames)
    if not any_ball_detected and (args.kick_frame is None or args.contact_frame is None):
        raise ValueError("Ball was never detected in this clip. Pass --kick_frame and --contact_frame manually.")

    comp_positions = [f["comp_xy"] for f in frames]
    raw_positions  = [f["raw_xy"] for f in frames]
    is_predicted   = [f["is_predicted"] for f in frames]
    players        = [f["players"] for f in frames]

    detector_cfg = EventTimingDetector(
        baseline_window=args.baseline_window,
        stationary_thresh=args.stationary_thresh,
        kick_sigma=args.kick_sigma,
        persistence_frames=args.persistence_frames,
        max_gap_frames=args.max_gap_frames,
        corner_margin_frac=args.corner_margin_frac,
        ballistic_fit_frames=args.ballistic_fit_frames,
        ballistic_fit_window_frames=args.ballistic_fit_window_frames,
        residual_thresh_px=args.residual_thresh_px,
        contact_proximity_px=args.contact_proximity_px,
        contact_search_frames=args.contact_search_frames,
        contact_persistence_frames=args.contact_persistence_frames,
        reversal_angle_thresh_deg=args.reversal_angle_thresh_deg,
        min_flight_speed_px=args.min_flight_speed_px,
        max_segment_gap_frames=args.max_segment_gap_frames,
        arc_break_residual_px=args.arc_break_residual_px,
        contact_max_gap_frames=args.contact_max_gap_frames,
        contact_min_score=args.contact_min_score,
        motion_spike_weight=args.motion_spike_weight,
    )

    kicker_id_finder = lambda kf: find_kicker(frames, kf, args.kicker_window)["tracker_id"]

    if args.kick_frame is not None and args.contact_frame is not None:
        kick_frame, kick_confidence = args.kick_frame, None
        contact_frame, contact_confidence, contact_tier, contact_window = args.contact_frame, None, None, None
        kicker_info = find_kicker(frames, kick_frame, args.kicker_window)
        kicker_id = kicker_info["tracker_id"]
    else:
        auto = detector_cfg.run(
            comp_positions, raw_positions, is_predicted, players,
            kicker_id_finder, frame_size=frame_size,
        )
        if args.kick_frame is None and auto["kick_frame"] is None:
            raise ValueError(
                "Could not confidently detect a kick frame -- no sustained speed spike found after "
                "the resting baseline. Pass --kick_frame manually rather than trust a guess."
            )
        kick_frame = args.kick_frame if args.kick_frame is not None else auto["kick_frame"]
        kick_confidence = None if args.kick_frame is not None else auto["kick_confidence"]
        kicker_info = find_kicker(frames, kick_frame, args.kicker_window)
        kicker_id = kicker_info["tracker_id"]

        if args.contact_frame is not None:
            contact_frame, contact_confidence, contact_tier, contact_window = args.contact_frame, None, None, None
        elif args.kick_frame is not None:
            contact = detector_cfg.detect_contact(comp_positions, raw_positions, is_predicted, kick_frame, players, kicker_id)
            contact_frame, contact_confidence = contact["frame"], contact["confidence"]
            contact_tier, contact_window = contact["tier"], contact["window"]
        else:
            contact_frame, contact_confidence = auto["contact_frame"], auto["contact_confidence"]
            contact_tier, contact_window = auto["contact_tier"], auto["contact_window"]

    contact_tracker_id = None
    if contact_frame is not None and 0 <= contact_frame < n:
        contact_tracker_id, _ = _nearest_player(frames[contact_frame])

    return {
        "video_path": video_path,
        "fps": fps,
        "frame_size": frame_size,
        "n_frames": n,
        "frames": frames,
        "cam_motion_series": cam_motion_series,
        "ball_candidates": ball_candidates,
        "cluster_summary": cluster_summary,
        "decoy_ids": decoy_ids,
        "gate_stats": gate_stats,
        "kick_frame": kick_frame,
        "kick_confidence": kick_confidence,
        "contact_frame": contact_frame,
        "contact_confidence": contact_confidence,
        "contact_tier": contact_tier,
        "contact_window": contact_window,
        "kicker_id": kicker_id,
        "kicker_box": kicker_info["box"],
        "contact_tracker_id": contact_tracker_id,
        "overridden": {
            "kick": args.kick_frame is not None,
            "contact": args.contact_frame is not None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        device = _get_device(args.device)
        print(f"[INFO] Running detection+tracking over {args.input} (device={device}, motion_model={args.motion_model}) ...")
        result = run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    video_path = result["video_path"]
    fps = result["fps"]
    n = result["n_frames"]
    kick_frame = result["kick_frame"]
    contact_frame = result["contact_frame"]

    print(f"[INFO] Processed {n} frames at {fps:.2f} fps.")
    print(f"[INFO] Gate rejections: {result['gate_stats']}")
    decoys = [info for cid, info in result["cluster_summary"].items() if cid in result["decoy_ids"]]
    print(f"[INFO] Rejected {len(decoys)} static decoy cluster(s): "
          + ", ".join(f"centroid={d['centroid']} frames={d['frame_count']}" for d in decoys) if decoys else "[INFO] No static decoys rejected.")

    conf_str = f", confidence={result['kick_confidence']}" if result["kick_confidence"] is not None else " (manual)"
    print(f"[INFO] Kick frame = {kick_frame} ({kick_frame / fps:.2f}s){conf_str}")
    print(f"[INFO] Kicker tracker_id = {result['kicker_id']}")

    contact_tier = result["contact_tier"]
    contact_window = result["contact_window"]
    if contact_frame is None:
        if contact_tier == 3 and contact_window is not None:
            w_lo, w_hi = contact_window
            print(f"[INFO] Contact: Tier 3 (abstain) -- ball isn't confidently trackable through the touch. "
                  f"Window = frames {w_lo}-{w_hi} ({w_lo/fps:.2f}s-{w_hi/fps:.2f}s), confidence={result['contact_confidence']}.")
        else:
            print("[WARN] Could not auto-detect a first-contact frame; pass --contact_frame to override.")
    else:
        conf_str = f", confidence={result['contact_confidence']}" if result["contact_confidence"] is not None else " (manual)"
        tier_str = f", tier={contact_tier}" if contact_tier is not None else ""
        print(f"[INFO] Contact frame = {contact_frame} ({contact_frame / fps:.2f}s){conf_str}{tier_str}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    decoy_ids = result["decoy_ids"]
    ball_candidates = result["ball_candidates"]

    def _read_frame(idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read frame {idx} from {video_path}")
        return f

    kicker_markers = [(result["kicker_box"], "KICKER", (0, 215, 255))] if result["kicker_box"] is not None else []

    kick_path = output_dir / f"{stem}_kick_snapshot.png"
    kick_bbox, kick_conf = find_ball_detection_at(result["frames"][kick_frame], ball_candidates[kick_frame], decoy_ids)
    kick_img = draw_annotated_snapshot(
        _read_frame(kick_frame), kick_frame, fps, "KICK",
        result["frames"][kick_frame]["player_boxes"], result["frames"][kick_frame]["raw_xy"],
        kick_bbox, kick_conf, None, "",
        include_label=not args.no_label,
        manual_markers=kicker_markers,
    )
    cv2.imwrite(str(kick_path), kick_img)
    print(f"[OK] Saved kick snapshot -> {kick_path}")
    if not kicker_markers:
        print("[WARN] Could not identify the corner-taker near the ball -- no KICKER marker drawn.", file=sys.stderr)

    if args.kick_window_frames > 0:
        kick_window_dir = output_dir / "kick_window"
        kick_window_dir.mkdir(parents=True, exist_ok=True)
        lo = max(0, kick_frame - args.kick_window_frames)
        hi = min(n - 1, kick_frame + args.kick_window_frames)
        pad = len(str(hi))
        for idx in range(lo, hi + 1):
            f_data = result["frames"][idx]
            bbox, conf = find_ball_detection_at(f_data, ball_candidates[idx], decoy_ids)
            label = "KICK" if idx == kick_frame else f"KICK{idx - kick_frame:+d}"
            img = draw_annotated_snapshot(
                _read_frame(idx), idx, fps, label,
                f_data["player_boxes"], f_data["raw_xy"], bbox, conf, None, "",
                include_label=not args.no_label,
                manual_markers=kicker_markers if idx == kick_frame else [],
            )
            out_path = kick_window_dir / f"{stem}_kick_window_{idx:0{pad}d}.png"
            cv2.imwrite(str(out_path), img)
        print(f"[OK] Saved {hi - lo + 1} kick-window frames ({lo}-{hi}) -> {kick_window_dir}")

    contact_path = None
    contact_bbox = contact_conf = None
    if contact_frame is not None:
        contact_path = output_dir / f"{stem}_first_contact_snapshot.png"
        contact_bbox, contact_conf = find_ball_detection_at(
            result["frames"][contact_frame], ball_candidates[contact_frame], decoy_ids,
        )
        contact_img = draw_annotated_snapshot(
            _read_frame(contact_frame), contact_frame, fps, "FIRST CONTACT",
            result["frames"][contact_frame]["player_boxes"], result["frames"][contact_frame]["raw_xy"],
            contact_bbox, contact_conf, result["contact_tracker_id"], "CONTACT",
            include_label=not args.no_label,
        )
        cv2.imwrite(str(contact_path), contact_img)
        print(f"[OK] Saved first-contact snapshot -> {contact_path}")

    if args.dump_all_frames:
        all_frames_dir = output_dir / "all_frames"
        all_frames_dir.mkdir(parents=True, exist_ok=True)
        highlights: dict[int, tuple[str, tuple[int, int, int]]] = {}
        if result["contact_tracker_id"] is not None:
            highlights[result["contact_tracker_id"]] = ("CONTACT", (255, 0, 255))

        cap = cv2.VideoCapture(str(video_path))
        pad = len(str(n - 1))
        for idx in range(n):
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] Could not read frame {idx} while dumping all frames.", file=sys.stderr)
                break
            f_data = result["frames"][idx]
            bbox, conf = find_ball_detection_at(f_data, ball_candidates[idx], decoy_ids)
            if idx == kick_frame:
                label = "KICK"
            elif contact_frame is not None and idx == contact_frame:
                label = "FIRST CONTACT"
            else:
                label = "FRAME"
            markers = kicker_markers if idx == kick_frame else []
            img = draw_annotated_snapshot(
                frame, idx, fps, label,
                f_data["player_boxes"], f_data["raw_xy"], bbox, conf,
                highlight_id=None, highlight_role="",
                include_label=not args.no_label,
                extra_highlights=highlights,
                manual_markers=markers,
            )
            out_path = all_frames_dir / f"{stem}_frame_{idx:0{pad}d}.png"
            cv2.imwrite(str(out_path), img)
        cap.release()
        print(f"[OK] Saved {n} annotated frames -> {all_frames_dir}")

    window_center = contact_frame
    window_bounds = None
    if window_center is None and contact_window is not None:
        window_bounds = contact_window

    if args.contact_window_frames > 0 and (window_center is not None or window_bounds is not None):
        window_dir = output_dir / "contact_window"
        window_dir.mkdir(parents=True, exist_ok=True)
        highlights: dict[int, tuple[str, tuple[int, int, int]]] = {}
        if result["contact_tracker_id"] is not None:
            highlights[result["contact_tracker_id"]] = ("CONTACT", (255, 0, 255))

        if window_center is not None:
            lo = max(0, window_center - args.contact_window_frames)
            hi = min(n - 1, window_center + args.contact_window_frames)
        else:
            lo = max(0, window_bounds[0] - args.contact_window_frames)
            hi = min(n - 1, window_bounds[1] + args.contact_window_frames)
        pad = len(str(hi))
        for idx in range(lo, hi + 1):
            f_data = result["frames"][idx]
            bbox, conf = find_ball_detection_at(f_data, ball_candidates[idx], decoy_ids)
            if window_center is not None:
                label = "FIRST CONTACT" if idx == window_center else f"CONTACT{idx - window_center:+d}"
            elif window_bounds is not None and window_bounds[0] <= idx <= window_bounds[1]:
                label = "ABSTAIN WINDOW"
            else:
                label = "FRAME"
            img = draw_annotated_snapshot(
                _read_frame(idx), idx, fps, label,
                f_data["player_boxes"], f_data["raw_xy"], bbox, conf,
                highlight_id=None, highlight_role="",
                include_label=not args.no_label,
                extra_highlights=highlights,
            )
            out_path = window_dir / f"{stem}_contact_window_{idx:0{pad}d}.png"
            cv2.imwrite(str(out_path), img)
        print(f"[OK] Saved {hi - lo + 1} contact-window frames ({lo}-{hi}) -> {window_dir}")

    debug_plot_path = output_dir / "ball_speed_debug.png"
    save_debug_plot(result["frames"], result["cam_motion_series"], kick_frame, contact_frame, debug_plot_path)
    print(f"[OK] Saved debug plot -> {debug_plot_path}")

    cam_pan_series = [float(np.hypot(dx, dy)) for dx, dy in result["cam_motion_series"]]
    v_pixel_series = [0.0] + [
        float(np.hypot(result["frames"][i]["raw_xy"][0] - result["frames"][i - 1]["raw_xy"][0],
                        result["frames"][i]["raw_xy"][1] - result["frames"][i - 1]["raw_xy"][1]))
        for i in range(1, n)
    ]
    v_comp_series = compensated_speed_series([f["comp_xy"] for f in result["frames"]]).tolist()

    metadata = {
        "video": str(video_path),
        "fps": fps,
        "total_frames": n,
        "motion_model": args.motion_model,
        "kick_frame": kick_frame,
        "kick_confidence": result["kick_confidence"],
        "kick_frame_time_s": round(kick_frame / fps, 3),
        "contact_frame": contact_frame,
        "contact_confidence": result["contact_confidence"],
        "contact_frame_time_s": round(contact_frame / fps, 3) if contact_frame is not None else None,
        "contact_tier": contact_tier,
        "contact_window": contact_window,
        "contact_window_time_s": (
            [round(contact_window[0] / fps, 3), round(contact_window[1] / fps, 3)]
            if contact_window is not None else None
        ),
        "kicker_tracker_id": result["kicker_id"],
        "kicker_box": list(result["kicker_box"]) if result["kicker_box"] is not None else None,
        "contact_tracker_id": result["contact_tracker_id"],
        "kick_ball_detection": {
            "bbox": list(kick_bbox) if kick_bbox is not None else None,
            "confidence": kick_conf,
            "is_predicted": result["frames"][kick_frame]["is_predicted"],
        },
        "contact_ball_detection": {
            "bbox": list(contact_bbox) if contact_bbox is not None else None,
            "confidence": contact_conf,
            "is_predicted": result["frames"][contact_frame]["is_predicted"] if contact_frame is not None else None,
        },
        "players_detected_at_kick": len(result["frames"][kick_frame]["player_boxes"]),
        "players_detected_at_contact": (
            len(result["frames"][contact_frame]["player_boxes"]) if contact_frame is not None else None
        ),
        "camera_pan_series": cam_pan_series,
        "v_pixel_series": v_pixel_series,
        "v_comp_series": v_comp_series,
        "rejected_static_decoys": [
            {"cluster_id": cid, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in info.items()}}
            for cid, info in result["cluster_summary"].items() if cid in result["decoy_ids"]
        ],
        "gate_stats": result["gate_stats"],
        "overridden": result["overridden"],
    }
    def _json_default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    metadata_path = output_dir / "snapshot_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_default))
    print(f"[OK] Saved metadata -> {metadata_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
