# Football Tracker — App

This folder contains a Streamlit dashboard for the Football Tracking pipeline.

Quick start (macOS / zsh):

1. Create and activate a virtual environment (Python 3.10+):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install requirements:

```bash
pip install -r requirements.txt
# also install streamlit separately if not in requirements
pip install streamlit
```

3. Ensure you have model weights and at least one video in `data/raw/`:

- Model path used by the app: `models/best.pt` (configured in `app/config.py`).
- The repository also contains `yolov8m_fixed.pt` for the `main.py` pipeline.

4. Run the Streamlit app from the project root:

```bash
streamlit run app/Home.py
```

Notes and caveats
- The analysis page calls the project's `main.py` pipeline. That script is heavy (OpenCV, YOLO, ByteTrack) and may be slow on CPU.
- Make sure `requirements.txt` is installed and the YOLO weights are present under `models/`.
- If you only want to run the offline `main.py` pipeline without the dashboard, use:

```bash
python main.py --input data/raw/your_video.mp4 --output_dir results
```

If you want, I can also add a lightweight wrapper script to run the Streamlit app with a check for required models and packages.
