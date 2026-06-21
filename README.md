# Football Tracker — Match Video Analytics Pipeline

A Streamlit-based football analytics platform that turns raw match video into structured data: player tracking, team classification, possession analysis, speed estimation, and annotated video output.

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
| **AI Analyst (Gemini)** | Natural-language match reports + grounded Q&A chat — see [docs/AI_ANALYST.md](docs/AI_ANALYST.md) |

---

## Quick Start

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
4. **AI Analyst** — generate a Gemini match report or ask questions about the match (requires `GEMINI_API_KEY`; see [docs/AI_ANALYST.md](docs/AI_ANALYST.md))

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

```bash
python main.py --input data/raw/match.mp4 --output_dir results --max_frames 0
```

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Path to input video |
| `--output_dir` | `results` | Directory for outputs |
| `--max_frames` | `0` (all) | Limit frames processed (0 = full video) |

---

## Deployment

### Streamlit Community Cloud

1. Push to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set **Main file path** to `dashboard/Home.py`
4. Add model weights via Streamlit Secrets or a download script



---

## Requirements

- Python 3.10+
- PyTorch 2.2+ (CUDA optional but recommended)
- See `requirements.txt` for full list

---

## License

MIT
