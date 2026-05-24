"""
Fine-tunes a YOLO model on the SoccerNet-GSR dataset.

Requires: pip install ultralytics

Usage:
    python scripts/train_yolo.py
    python scripts/train_yolo.py --model yolo11n.pt --epochs 50 --batch 16
    python scripts/train_yolo.py --resume runs/finetune/soccernet_team_classifier/weights/last.pt
    python scripts/train_yolo.py --dry-run   # validate config without training

The trained model will be at: runs/finetune/soccernet_team_classifier/weights/best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET_YAML = Path("datasets/soccernet_yolo/dataset.yaml")
DEFAULT_MODEL = "yolo11s.pt"
OUTPUT_DIR = Path("runs/finetune")
PROJECT_NAME = "soccernet_team_classifier"


def _gpu_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def run_training(args) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics: pip install ultralytics") from exc

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATASET_YAML}\n"
            f"Run phase 3 first: python scripts/convert_annotations.py"
        )

    if args.resume:
        log.info("Resuming training from: %s", args.resume)
        model = YOLO(str(args.resume))
    else:
        log.info("Loading base model: %s", args.model)
        model = YOLO(args.model)

    if args.dry_run:
        log.info("Dry run: validating config only (no training)")
        model.val(data=str(DATASET_YAML), imgsz=args.imgsz, device=args.device)
        return

    log.info(
        "Starting fine-tune:\n"
        "  Model:   %s\n"
        "  Dataset: %s\n"
        "  Epochs:  %s\n"
        "  Batch:   %s\n"
        "  Imgsz:   %s\n"
        "  Device:  %s\n"
        "  Freeze:  %s\n",
        args.model,
        DATASET_YAML,
        args.epochs,
        args.batch,
        args.imgsz,
        args.device,
        args.freeze,
    )

    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(OUTPUT_DIR),
        name=PROJECT_NAME,
        exist_ok=args.resume is not None,
        freeze=args.freeze,
        cls=2.0,
        box=7.5,
        dfl=1.5,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.0,
        mosaic=0.5,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        warmup_epochs=3,
        save_period=5,
        val=True,
        plots=True,
        verbose=True,
        patience=50,
    )

    best_path = OUTPUT_DIR / PROJECT_NAME / "weights" / "best.pt"
    last_path = OUTPUT_DIR / PROJECT_NAME / "weights" / "last.pt"
    results_png = OUTPUT_DIR / PROJECT_NAME / "results.png"

    log.info("\nTraining complete.")
    log.info("Best model: %s", best_path)
    log.info("Last model: %s", last_path)
    log.info("Results:    %s", results_png)

    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        log.info("\nFinal metrics:")
        for key, value in metrics.items():
            try:
                log.info("  %s: %.4f", key, float(value))
            except (TypeError, ValueError):
                log.info("  %s: %s", key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on SoccerNet-GSR")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model weights (e.g. yolo11s.pt, yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size. Reduce if OOM. Use -1 for auto.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input image size. 1280 for broadcast, 640 for faster training")
    parser.add_argument(
        "--device",
        default="0" if _gpu_available() else "cpu",
        help="Device: '0' for GPU 0, 'cpu', '0,1' for multi-GPU",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint (.pt file)")
    parser.add_argument("--freeze", type=int, default=10, help="Freeze this many backbone layers (use 0 when resuming full fine-tune)")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
