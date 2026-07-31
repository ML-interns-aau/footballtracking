# Football Tracker — Match Video Analytics Pipeline

A Streamlit-based football analytics platform that turns raw match video : player tracking, team classification, possession analysis, speed estimation, and annotated video output.

---

## Features

| Feature | Description |
|---|---|
| **Player Detection** | YOLOv8 object detection (players + ball) |
| **Multi-object Tracking** | ByteTrack with camera-motion compensation |
| **Team Classification** | HSV jersey-colour clustering (KMeans) |
| **Ball Tracking** | Kalman-filter gap filling with trail visualisation |
| **Speed Estimation** | Homography-based pitch mapping → km/h |
| **Possession Analysis** | Per-team and per-player possession percentages |
| **Annotated Video** | Full output video with bounding boxes, trails, HUD |
| **CSV / JSON Export** | Player summary, tracking data, pipeline summary |
| **GPU Acceleration** | Automatic CUDA detection; falls back to CPU |
| **AI Analyst (Gemini + Groq)** | Multi-provider natural-language match reports + grounded Q&A chat, with provider comparison — see [docs/AI_ANALYST.md](docs/AI_ANALYST.md) |
| **Corner-Kick Timing** | Automatic kick-moment and first-contact frame extraction from a corner-kick clip, camera-motion compensated with decoy-aware ball tracking — see [Corner-Kick Timing](#corner-kick-timing) below |

---


### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** install the CUDA-enabled PyTorch build first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Add model weights

Place your trained YOLO model at:
```
yolov8m_fixed.pt          # project root (preferred)
# or
models/best.pt            # fallback
```

### 3. Run the app

```bash
streamlit run app/Home.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Upload** — drag-and-drop a match video (MP4, AVI, MOV, MKV) or select one from `data/raw/`
2. **Analysis** — click **Run Full Pipeline**; a live progress bar tracks processing
3. **Results** — view possession charts, player stats, speed analysis, and download outputs
4. **AI Analyst** — generate a match report, ask questions, or compare providers (requires `GEMINI_API_KEY` and/or `GROQ_API_KEY`; see [docs/AI_ANALYST.md](docs/AI_ANALYST.md))

---

## Project Structure

```text
football_tracking_project/
├── app/                         # Streamlit UI code (Frontend)
│   ├── Home.py                  # Entry point
│   ├── config.py                # Paths and defaults
│   ├── utils.py                 # Shared UI components
│   └── pages/                   # Page modules (upload, analysis, results)
├── src/                         # Computer vision & analytics (Backend)
│   ├── engine/                  # Detectors, trackers, and core computer vision
│   ├── analytics/               # Events, heatmaps, and speed estimation
│   ├── exporters/               # JSON/CSV formatting and outputs
│   └── visualization/           # Frame annotation
├── main.py                      # Clean pipeline entry point
├── tools/
│   ├── extract_corner_snapshots.py  # Corner-kick timing: auto kick/contact frame extraction
│   └── validate.py                  # Ground-truth accuracy validation for the above
├── models/
│   └── yolov8m_fixed.pt         # YOLO weights
├── data/
│   ├── raw/                     # Input videos
│   ├── processed/               # Preprocessed videos
│   ├── annotations/
│   └── insights/                # Generated CSVs + JSON
├── results/                     # Pipeline outputs (annotated video, CSVs)
├── configs/
│   └── config.yaml
├── requirements.txt
└── .streamlit/
    └── config.toml              # Streamlit server config
```

---

## CLI Usage

Run the full tracking pipeline against a video from the terminal (no
Streamlit UI needed) and get all results written straight to
`--output_dir`:

```bash
python main.py --input data/raw/match.mp4 --output_dir results --max_frames 0
```

This writes, directly under `--output_dir`:

| File | Contents |
|---|---|
| `annotated_football_analysis.mp4` | Full video with bounding boxes, trails, HUD |
| `tracking_output.csv` | Per-frame per-player tracking data |
| `analytics.csv` / `analytics.json` | Possession, speed, and event analytics |
| `player_summary.csv` | Per-player summary stats |
| `possession_summary.csv` | Per-team possession breakdown |
| `team_0_heatmap.png` / `team_1_heatmap.png` | Per-team position heatmaps |
| `metadata.json` | Run metadata (video, model, timings) |

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Path to input video |
| `--output_dir` | `results` | Directory for outputs |
| `--max_frames` | `0` (all) | Limit frames processed (0 = full video) |
| `--target_fps` | see `app/config.py` | Frame rate to process at (subsamples if lower than source) |
| `--resize_width` | see `app/config.py` | Resize frame width before detection |
| `--conf` / `--iou` / `--imgsz` | see `app/config.py` | YOLO detection thresholds/input size |
| `--device` | auto | `cuda` or `cpu`; auto-detects if omitted |
| `--model_path` | see `app/config.py` | Path to YOLO weights |
| `--game_id` | auto-generated | Subfolder name under `--output_dir` for this run |
| `--match-id` / `--home-team` / `--away-team` | placeholders | Metadata recorded in `metadata.json` |

---

## Corner-Kick Timing

Extracts two key frames from a corner-kick clip: the moment the ball is
struck (**kick**) and the frame of first contact by another player
afterward (**contact**). Detection/tracking runs entirely in a
camera-motion-compensated coordinate space so pan doesn't corrupt the
timing, with decoy-aware ball clustering to reject sideline balls and
sponsor-board false positives.

```bash
python tools/extract_corner_snapshots.py \
    --input clip_1.mp4 --output_dir results/corner_snapshots \
    --model models/yolo11l.pt --ball_weights models/soccana_yolo11n.pt

# Also dump annotated frames around the kick/contact for manual review:
python tools/extract_corner_snapshots.py --input clip_1.mp4 --output_dir results/corner_snapshots \
    --model models/yolo11l.pt --ball_weights models/soccana_yolo11n.pt \
    --kick-window-frames 15 --contact-window-frames 15

# Dump every frame of the clip, annotated:
python tools/extract_corner_snapshots.py --input clip_1.mp4 --output_dir results/corner_snapshots \
    --model models/yolo11l.pt --ball_weights models/soccana_yolo11n.pt --dump-all-frames

# Override an auto-detected frame if it looks wrong on ball_speed_debug.png
# (or after confirming the true frame from a --*-window-frames/--dump-all-frames dump):
python tools/extract_corner_snapshots.py --input clip_1.mp4 \
    --kick_frame 42 --contact_frame 57
```

Outputs `<clip>_kick_snapshot.png`, `<clip>_first_contact_snapshot.png`
(when contact resolves to an exact frame), a debug plot of ball speed vs.
frame index, and `snapshot_metadata.json` with the thresholds, tiers, and
confidence used. A pass1+2 result cache (`.cache/pass12/`, git-ignored)
skips re-running detection when only the event-timing parameters change,
since detection is the expensive part by a wide margin.

**Contact detection is tiered**, never emitting a confident frame the
evidence doesn't support:

| Tier | Meaning | Output |
|---|---|---|
| 1 | Ball visible on both sides of the touch, no occlusion gap | exact frame, high confidence |
| 2 | Ball visible before/after a short occlusion gap bracketing the touch | exact frame (gap start), medium confidence |
| 3 | Ball undetectable through the touch (small/blurred, or a crowded box with no single clean signature) | `contact_frame: null`, plus a `contact_window` for manual confirmation |

Kick detection can likewise abstain (`kick_frame: null`) if no sustained
speed spike is found after the resting baseline, rather than guessing the
loudest noise in the clip.

Manual-review helpers, both git-ignored, saved alongside the snapshots:
- `--kick-window-frames N` / `--contact-window-frames N` — dump N annotated
  frames on either side of the kick/contact pick (or the Tier 3 window) into
  `kick_window/` / `contact_window/`, for scrubbing by eye when a pick looks
  off.
- `--dump-all-frames` — annotate and save every frame of the clip into
  `all_frames/`.

If the ball is too small/distant to detect reliably near the corner arc,
`--tile-ball-detection` (runs detection on overlapping crops instead of the
full downscaled frame) and a lower `--ball-conf` can recover it — verify
against `tools/validate.py` afterward, since lowering the threshold trades
recall for false-positive risk.

Validate against ground truth:

```bash
python tools/validate.py
```

