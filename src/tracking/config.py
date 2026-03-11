import os
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


def _load_dotenv_if_present() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv_if_present()


class Config:
    INPUT_VIDEO = "data/raw/114.mp4"

    # Model settings
    DETECTOR_BACKEND = os.getenv(
        "DETECTOR_BACKEND",
        "roboflow" if os.getenv("ROBOFLOW_API_KEY") else "ultralytics",
    )
    MODEL_PATH = "yolov8n.pt"
    ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "football-players-detection-3zvbc/20")
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
    ROBOFLOW_TIMEOUT_SECONDS = float(os.getenv("ROBOFLOW_TIMEOUT_SECONDS", "20"))
    ROBOFLOW_CONNECT_TIMEOUT_SECONDS = float(os.getenv("ROBOFLOW_CONNECT_TIMEOUT_SECONDS", "5"))
    ROBOFLOW_READ_TIMEOUT_SECONDS = float(os.getenv("ROBOFLOW_READ_TIMEOUT_SECONDS", "20"))
    ROBOFLOW_MAX_RETRIES = int(os.getenv("ROBOFLOW_MAX_RETRIES", "2"))
    ROBOFLOW_RETRY_BACKOFF_SECONDS = float(os.getenv("ROBOFLOW_RETRY_BACKOFF_SECONDS", "0.7"))
    ROBOFLOW_OVERLAP = float(os.getenv("ROBOFLOW_OVERLAP", "30"))
    ROBOFLOW_MAX_INFER_SIZE = int(os.getenv("ROBOFLOW_MAX_INFER_SIZE", "640"))
    ROBOFLOW_JPEG_QUALITY = int(os.getenv("ROBOFLOW_JPEG_QUALITY", "75"))
    ROBOFLOW_INFER_EVERY_N_FRAMES = int(os.getenv("ROBOFLOW_INFER_EVERY_N_FRAMES", "2"))
    CONFIDENCE = 0.35
    IOU_THRESHOLD = 0.45
    DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    TRACKER_TYPE = "bytetrack.yaml"
    INFERENCE_IMGSZ = 960
    MAX_DETECTIONS = 80
    USE_HALF_PRECISION = True
    CLASS_FILTER = [0, 32]  # person, sports ball (if present in model)

    # Class mappings
    CLASS_PERSON = 0
    BALL_CLASS_IDS = {32}  # sports-ball on COCO models

    # Frame processing
    TARGET_SIZE = (1280, 720)
    ENABLE_FIELD_WARP = True

    # Jersey classification (BGR anchors)
    TEAM1_COLOR = (220, 60, 60)
    TEAM2_COLOR = (60, 60, 220)
    GOALKEEPER_COLOR = (0, 220, 220)
    REFEREE_COLOR = (30, 30, 30)

    # Field keypoints (32 characteristic points in normalized field space)
    FIELD_KEYPOINT_TEMPLATE = [
        (0.00, 0.00), (0.25, 0.00), (0.50, 0.00), (0.75, 0.00), (1.00, 0.00),
        (0.00, 0.15), (0.17, 0.15), (0.33, 0.15), (0.50, 0.15), (0.67, 0.15), (0.83, 0.15), (1.00, 0.15),
        (0.00, 0.33), (0.20, 0.33), (0.40, 0.33), (0.60, 0.33), (0.80, 0.33), (1.00, 0.33),
        (0.00, 0.50), (0.30, 0.50), (0.50, 0.50), (0.70, 0.50), (1.00, 0.50),
        (0.00, 0.67), (0.20, 0.67), (0.40, 0.67), (0.60, 0.67), (0.80, 0.67), (1.00, 0.67),
        (0.00, 1.00), (0.50, 1.00), (1.00, 1.00),
    ]

    # Output/persistence settings
    OUTPUT_ROOT_DIR = "data/processed/tracking"
    SAVE_TRACKING_EXPORTS = True
    SHOW_PREVIEW = False
