# Quick Start Guide

## New Developer Onboarding (5 minutes)

### 1. Prerequisites

Ensure you have Python 3.10+ installed.

### 2. Setup Environment

```bash
# Clone repository
git clone <repo-url>
cd footballtracking

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Pipeline (Backend)

To run the tracking and event detection pipeline on a video:

```bash
python main.py --input data/raw/video.mp4 --output_dir results/ --game_id game_001
```

For quick testing, you can limit the number of frames to process:
```bash
python main.py --input data/raw/video.mp4 --output_dir results/ --max_frames 250
```

### 4. Running the Dashboard (Frontend)

To start the Streamlit UI and view the analytics:

```bash
streamlit run app/Home.py
```
**App will open at:** http://localhost:8501

---

## Project Structure (What You Need to Know)

```
footballtracking/
├── app/                    # Streamlit UI code (Frontend)
│   ├── Home.py            # Entry point
│   ├── pages/             # Page modules
│   └── config.py          # UI config
├── src/                   # Computer vision & analytics (Backend)
│   ├── engine/            # Detectors, trackers, and core computer vision
│   ├── analytics/         # Events, heatmaps, and speed estimation
│   ├── exporters/         # JSON/CSV formatting and outputs
│   └── visualization/     # Frame annotation
├── main.py                # Clean pipeline entry point
└── requirements.txt       # Python dependencies
```

---

## Daily Development Workflow

1. **Activate Environment:** `source .venv/bin/activate`
2. **Run Pipeline:** `python main.py --input <path> --output_dir results`
3. **Run UI:** `streamlit run app/Home.py`
