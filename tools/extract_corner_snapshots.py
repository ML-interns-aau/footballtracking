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


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _get_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Pass 1: detection + player tracking + camera motion + ball-candidate clustering
# ---------------------------------------------------------------------------

def run_pass_one(video_path: Path, args, device: str):
    detector = FootballDetector(
        model_path=args.model, conf=args.conf, iou=args.iou, device=device,
        imgsz=args.imgsz,
        use_pitch_gate=args.enable_pitch_gate,
        use_led_gate=args.enable_led_gate,
        ball_weights=args.ball_weights,
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
    per_frame_ball_candidates: list[list[dict]] = []  # each: {"raw_xy", "comp_xy", "confidence", "cluster_id"}
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
        if detections.class_id is not None:
            for i in np.where(detections.class_id == 32)[0]:
                x1, y1, x2, y2 = detections.xyxy[i]
                rx, ry = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                conf = float(detections.confidence[i]) if detections.confidence is not None else 1.0
                comp_xy = (rx - cum_dx, ry - cum_dy)
                cid = clusterer.add(idx, comp_xy, conf)
                candidates.append({"raw_xy": (rx, ry), "comp_xy": comp_xy, "confidence": conf, "cluster_id": cid})
        per_frame_ball_candidates.append(candidates)

        players: dict[int, tuple[float, float]] = {}
        for i in range(len(tracked)):
            class_id = int(tracked.class_id[i])  if tracked.class_id  is not None else -1
            tid      = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None
            if class_id != 0 or tid is None:
                continue
            x1, y1, x2, y2 = tracked.xyxy[i]
            players[tid] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        per_frame_players.append(players)

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
        "ball_candidates": per_frame_ball_candidates,
        "cam_motion_series": cam_motion_series,
        "cum_offset_series": cum_offset_series,
        "clusterer": clusterer,
        "gate_stats": dict(detector.last_gate_stats),
    }


# ---------------------------------------------------------------------------
# Pass 2: decoy-filtered candidate selection + compensated Kalman smoothing
# ---------------------------------------------------------------------------

def run_pass_two(pass1: dict, args) -> list[dict]:
    cluster_summary = pass1["clusterer"].classify()
    decoy_ids = {cid for cid, info in cluster_summary.items() if info["is_decoy"]}

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
            same_cluster = (
                [c for c in candidates if c["cluster_id"] == active_cluster_id]
                if active_cluster_id is not None and idx - active_last_seen <= args.candidate_continuity_ttl
                else []
            )
            best = max(same_cluster, key=lambda c: c["confidence"]) if same_cluster else \
                max(candidates, key=lambda c: c["confidence"])
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
        })

    cap.release()
    return frames, cluster_summary, decoy_ids


# ---------------------------------------------------------------------------
# Pass 1+2 cache -- detection/tracking/camera-motion/decoy-clustering is the
# expensive part (a full CPU pass over the clip); event_detector.py tuning
# only touches downstream args, so cache the pass1+2 output keyed on every arg
# that actually changes its result, and skip straight to event timing on a hit.
# ---------------------------------------------------------------------------

_CACHE_VERSION = 3  # bump when the cached tuple's shape OR pass1/2 logic changes

_PASS12_ARG_NAMES = [
    "model", "conf", "iou", "imgsz", "device", "motion_model", "ball_weights",
    "enable_pitch_gate", "enable_led_gate", "cluster_radius_px",
    "candidate_continuity_ttl", "decoy_early_frac", "decoy_late_frac",
    "decoy_min_frames", "decoy_min_span_frac", "max_missed", "diverge_px",
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


# ---------------------------------------------------------------------------
# Kicker identification (raw pixel proximity, same idea as before)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def draw_label(frame: np.ndarray, text: str) -> np.ndarray:
    frame = frame.copy()
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 12
    x, y = pad, pad + text_h
    cv2.rectangle(frame, (0, 0), (x + text_w + pad, y + baseline + pad), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def save_frame(video_path: Path, frame_idx: int, out_path: Path, label: str | None = None) -> None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    if label:
        frame = draw_label(frame, label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract kick-moment and first-contact snapshots from a corner-kick clip.")
    p.add_argument("--input",        required=True, help="Path to the corner-kick video clip.")
    p.add_argument("--output_dir",   default="results/corner_snapshots")
    p.add_argument("--model",        default="models/yolo11l.pt")
    # Empirically the small/blurry ball is barely detected at all at conf=0.30
    # (e.g. never once during clip_1's ~4s pre-kick resting period) -> lower
    # default so the kick/contact timing has real detections to work with.
    # The clustering/decoy-rejection layer is designed to absorb the resulting
    # extra noise candidates.
    p.add_argument("--conf",         type=float, default=0.15)
    p.add_argument("--iou",          type=float, default=0.40)
    # Native-resolution inference: clip_1 showed the resting corner-kick ball is
    # never detected at imgsz=960 (downsampling a 1280-wide frame loses the
    # small ball entirely) even at low confidence. 1280 keeps full resolution.
    p.add_argument("--imgsz",        type=int, default=1280)
    p.add_argument("--device",       default="auto", help="auto|cpu|cuda")

    p.add_argument("--motion_model", choices=["affine", "homography"], default="affine")
    p.add_argument("--ball_weights", default=None, help="Optional football/SoccerNet-finetuned ball checkpoint.")
    # Empirically, the HSV-based pitch/LED gates are unreliable in both directions
    # (too inclusive on stadiums with a green-toned perimeter track, too exclusive
    # right at the corner arc where grass segmentation is ambiguous) -> off by
    # default; the temporal-span decoy clustering in ball_tracker.py is the
    # primary defense. Kept as opt-in for footage where the color gates help.
    p.add_argument("--enable-pitch-gate", action="store_true")
    p.add_argument("--enable-led-gate",   action="store_true")

    p.add_argument("--kick_frame",    type=int, default=None, help="Manual override; skips kick-frame detection.")
    p.add_argument("--contact_frame", type=int, default=None, help="Manual override; skips contact-frame detection.")

    p.add_argument("--baseline-window",   type=int,   default=40)
    p.add_argument("--stationary-thresh", type=float, default=4.0)
    p.add_argument("--kick-sigma",        type=float, default=4.0)
    p.add_argument("--persistence-frames", type=int,  default=3)
    p.add_argument("--max-gap-frames",    type=int,   default=5)
    p.add_argument("--corner-margin-frac", type=float, default=0.22)

    p.add_argument("--kicker-window",        type=int,   default=10)
    # Quadratic ballistic fit needs >=4 points for a stable 3-parameter fit;
    # 8 gives it margin. Residual threshold lowered from the old linear-fit
    # default since a quadratic fit tracks real unobstructed flight tighter.
    p.add_argument("--ballistic-fit-frames", type=int,   default=8)
    p.add_argument("--ballistic-fit-window-frames", type=int, default=20,
                    help="How many frames past the kick the fit-gathering walk may range over, "
                         "bounded separately from --contact-search-frames so a long detection gap "
                         "can't make the fit consume real points near/after the actual contact.")
    p.add_argument("--residual-thresh-px",   type=float, default=15.0)
    p.add_argument("--contact-proximity-px", type=float, default=60.0)
    p.add_argument("--contact-search-frames", type=int,  default=45)
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

    p.add_argument("--cluster-radius-px",  type=float, default=60.0)
    p.add_argument("--candidate-continuity-ttl", type=int, default=15,
                    help="Frames to keep preferring the previously-active ball cluster before re-latching freely.")
    p.add_argument("--decoy-early-frac",   type=float, default=0.10)
    p.add_argument("--decoy-late-frac",    type=float, default=0.60)
    # A cluster only needs a few sparse sightings to prove it's a fixture if
    # they're spread across a long temporal span (a real ball moving that
    # rarely-and-that-slowly isn't a ball) -- 3 keeps single-frame noise out
    # without blocking the long_static_run rule from catching sparse decoys.
    p.add_argument("--decoy-min-frames",   type=int,   default=3)
    p.add_argument("--decoy-min-span-frac", type=float, default=0.40,
                    help="A static cluster spanning this fraction of the clip (regardless of when it starts) is a decoy.")

    p.add_argument("--max-missed",  type=int,   default=30)
    p.add_argument("--diverge-px",  type=float, default=250.0)

    p.add_argument("--no-label", action="store_true",
                    help="Don't burn frame#/time-stamp labels into the saved snapshot images.")

    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    """Runs the full pipeline and returns a result dict (no file I/O beyond
    what's needed internally). Used by both main() and tools/validate.py."""
    video_path = Path(args.input)
    if not video_path.exists():
        raise FileNotFoundError(f"Input not found: {video_path}")

    device = _get_device(args.device)
    frames, cluster_summary, decoy_ids, gate_stats, fps, frame_size, n, cam_motion_series, _ball_candidates = \
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
    )

    kicker_id_finder = lambda kf: find_kicker_id(frames, kf, args.kicker_window)  # noqa: E731

    if args.kick_frame is not None and args.contact_frame is not None:
        kick_frame, kick_confidence = args.kick_frame, None
        contact_frame, contact_confidence = args.contact_frame, None
        kicker_id = kicker_id_finder(kick_frame)
    else:
        auto = detector_cfg.run(
            comp_positions, raw_positions, is_predicted, players,
            kicker_id_finder, frame_size=frame_size,
        )
        kick_frame = args.kick_frame if args.kick_frame is not None else auto["kick_frame"]
        kick_confidence = None if args.kick_frame is not None else auto["kick_confidence"]
        kicker_id = auto["kicker_id"] if args.kick_frame is None else kicker_id_finder(kick_frame)

        if args.contact_frame is not None:
            contact_frame, contact_confidence = args.contact_frame, None
        elif args.kick_frame is not None:
            # Kick was overridden but contact wasn't -> re-run contact detection from the given kick.
            contact = detector_cfg.detect_contact(comp_positions, raw_positions, is_predicted, kick_frame, players, kicker_id)
            contact_frame, contact_confidence = contact["frame"], contact["confidence"]
        else:
            contact_frame, contact_confidence = auto["contact_frame"], auto["contact_confidence"]

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
        "cluster_summary": cluster_summary,
        "decoy_ids": decoy_ids,
        "gate_stats": gate_stats,
        "kick_frame": kick_frame,
        "kick_confidence": kick_confidence,
        "contact_frame": contact_frame,
        "contact_confidence": contact_confidence,
        "kicker_id": kicker_id,
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

    if contact_frame is None:
        print("[WARN] Could not auto-detect a first-contact frame; pass --contact_frame to override.")
    else:
        conf_str = f", confidence={result['contact_confidence']}" if result["contact_confidence"] is not None else " (manual)"
        print(f"[INFO] Contact frame = {contact_frame} ({contact_frame / fps:.2f}s){conf_str}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kick_label = None if args.no_label else f"KICK  |  frame {kick_frame}  |  t={kick_frame / fps:.2f}s"
    kick_path = output_dir / f"kick_frame_{kick_frame}.png"
    save_frame(video_path, kick_frame, kick_path, label=kick_label)
    print(f"[OK] Saved kick frame -> {kick_path}")

    contact_path = None
    if contact_frame is not None:
        contact_label = None if args.no_label else f"CONTACT  |  frame {contact_frame}  |  t={contact_frame / fps:.2f}s"
        contact_path = output_dir / f"contact_frame_{contact_frame}.png"
        save_frame(video_path, contact_frame, contact_path, label=contact_label)
        print(f"[OK] Saved contact frame -> {contact_path}")

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
        "kicker_tracker_id": result["kicker_id"],
        "contact_tracker_id": result["contact_tracker_id"],
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
