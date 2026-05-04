## Football Tracking Project

Computer vision pipeline for football match analysis using YOLOv8 detection, tracking, team classification, speed estimation, and pitch-space analytics.

### Features

- Player and ball detection with YOLOv8
- Multi-object tracking for player IDs across frames
- Team assignment from dominant jersey colors
- Camera motion compensation
- Pixel-to-pitch coordinate mapping
- Per-player speed and distance estimation
- Frame-by-frame analytics export (CSV and JSON)
- Team heatmap generation
- Annotated output video rendering

### Project Structure

```text
football_tracking_project/
	main.py
	download_model.py
	configs/
	dashboard/
	data/
	results/
	src/
```

### Requirements

- Python 3.10+
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

### Model Weights

The pipeline expects YOLO weights named `yolov8m_fixed.pt` in the project root.

You can download a compatible weight with:

```bash
python download_model.py
```

### Usage

Run the pipeline:

```bash
python main.py --input data/raw_videos/your_match.mp4 --output_dir results
```

Outputs are written to the selected output directory, including:

- `annotated_football_analysis.mp4`
- `analytics.csv`
- `analytics.json`
- team heatmaps (`team_0_heatmap.png`, `team_1_heatmap.png`)

### Notes

- The repository ignores large generated files and model binaries by default.
- If you want to version model files or result artifacts, remove the related patterns from `.gitignore`.

