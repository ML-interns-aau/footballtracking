# Phase 4 — Fine-Tuning YOLO on SoccerNet-GSR
# File: .github/instructions/04-yolo-finetune.instructions.md
# Apply to: scripts/train_yolo.py, runs/finetune/

## Goal

Fine-tune a YOLO model (YOLOv11n or YOLOv11s recommended) on the converted
SoccerNet-GSR dataset to directly predict 5 classes:

```
0: player_left
1: player_right
2: goalkeeper_left
3: goalkeeper_right
4: referee
```

The model learns team affiliation, role, and goalkeeper detection in a single
forward pass. This replaces the entire `TeamClassifier` color pipeline for
the detection stage.

---

## Model Selection

| Model | Params | Speed | mAP (baseline) | Recommended for |
|---|---|---|---|---|
| `yolo11n.pt` | 2.6M | Fastest | Good | CPU / resource-limited |
| `yolo11s.pt` | 9.4M | Fast | Better | **Default choice** |
| `yolo11m.pt` | 20M | Moderate | Best | GPU with ≥8GB VRAM |

Use `yolo11s.pt` unless GPU memory is limited. Start from the pretrained weights
— do NOT train from scratch. The pretrained weights encode general object detection
knowledge that transfers extremely well to player detection.

---

## Training Script

Create `scripts/train_yolo.py`:

```python
"""
Fine-tunes a YOLO model on the SoccerNet-GSR dataset.

Requires: pip install ultralytics

Usage:
    python scripts/train_yolo.py
    python scripts/train_yolo.py --model yolo11n.pt --epochs 50 --batch 16
    python scripts/train_yolo.py --resume runs/finetune/train/weights/last.pt
    python scripts/train_yolo.py --dry-run   # validate config without training

The trained model will be at: runs/finetune/train/weights/best.pt
"""
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET_YAML  = Path("datasets/soccernet_yolo/dataset.yaml")
DEFAULT_MODEL = "yolo11s.pt"
OUTPUT_DIR    = Path("runs/finetune")
PROJECT_NAME  = "soccernet_team_classifier"


def run_training(args) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATASET_YAML}\n"
            f"Run phase 3 first: python scripts/convert_annotations.py"
        )

    # Load model — either from pretrained weights or resume a checkpoint
    if args.resume:
        log.info(f"Resuming training from: {args.resume}")
        model = YOLO(args.resume)
    else:
        log.info(f"Loading base model: {args.model}")
        model = YOLO(args.model)

    if args.dry_run:
        log.info("Dry run: validating config only (no training)")
        model.val(data=str(DATASET_YAML), imgsz=args.imgsz)
        return

    log.info(
        f"Starting fine-tune:\n"
        f"  Model:   {args.model}\n"
        f"  Dataset: {DATASET_YAML}\n"
        f"  Epochs:  {args.epochs}\n"
        f"  Batch:   {args.batch}\n"
        f"  Imgsz:   {args.imgsz}\n"
        f"  Device:  {args.device}\n"
    )

    # ── Training call ──────────────────────────────────────────────────────
    # Key hyperparameters explained:
    #
    # freeze=10         : Freeze the first 10 backbone layers for the first phase.
    #                     This protects pretrained feature extractors while the
    #                     new 5-class head learns to converge. Remove after epoch ~20
    #                     to unfreeze the full model for refinement.
    #
    # cls=2.0           : Increase classification loss weight. Default is 0.5.
    #                     Our task requires accurate role/team discrimination,
    #                     not just detection, so we upweight cls loss.
    #
    # hsv_h, hsv_s, hsv_v : Color augmentation. Critical for generalizing across
    #                        different stadium lighting conditions and kit colors.
    #
    # fliplr=0.5        : Horizontal flip augmentation. Note: this SWAPS left/right
    #                     team labels. Handle in augmentation or disable if problematic.
    #                     See note below.
    #
    # degrees=5         : Slight rotation augmentation for camera angle variation.
    #
    # mosaic=0.5        : Mosaic augmentation (4 images combined). Helps with
    #                     scale variation for distant players.

    results = model.train(
        data        = str(DATASET_YAML),
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        device      = args.device,
        project     = str(OUTPUT_DIR),
        name        = PROJECT_NAME,
        exist_ok    = args.resume is not None,

        # Transfer learning: freeze backbone initially
        freeze      = 10,

        # Loss weights
        cls         = 2.0,    # higher classification loss weight
        box         = 7.5,    # keep box loss at default
        dfl         = 1.5,    # distribution focal loss

        # Augmentation (critical for lighting/kit generalization)
        hsv_h       = 0.02,   # hue shift ±2% (subtle color variation)
        hsv_s       = 0.7,    # saturation variation (lighting simulation)
        hsv_v       = 0.4,    # value/brightness variation (shadow simulation)
        degrees     = 5.0,    # rotation (camera angle variation)
        translate   = 0.1,    # translation
        scale       = 0.5,    # scale variation (near/far players)
        fliplr      = 0.0,    # DISABLED — horizontal flip swaps left/right team labels
                               # Enable only if you add label-swap logic in the dataloader
        mosaic      = 0.5,    # mosaic augmentation

        # Optimizer
        optimizer   = "AdamW",
        lr0         = 1e-3,   # initial learning rate
        lrf         = 0.01,   # final LR as fraction of lr0
        warmup_epochs = 3,

        # Saving & logging
        save_period = 5,      # save checkpoint every 5 epochs
        val         = True,
        plots       = True,
        verbose     = True,
    )

    log.info(f"\nTraining complete.")
    log.info(f"Best model: {OUTPUT_DIR}/{PROJECT_NAME}/weights/best.pt")
    log.info(f"Last model: {OUTPUT_DIR}/{PROJECT_NAME}/weights/last.pt")
    log.info(f"Results:    {OUTPUT_DIR}/{PROJECT_NAME}/results.png")

    # Print per-class mAP summary
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        log.info(f"\nFinal metrics:")
        for k, v in metrics.items():
            log.info(f"  {k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on SoccerNet-GSR")
    parser.add_argument("--model",   default=DEFAULT_MODEL,
                        help="Base model weights (e.g. yolo11s.pt, yolo11n.pt)")
    parser.add_argument("--epochs",  type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch",   type=int, default=16,
                        help="Batch size. Reduce if OOM. Use -1 for auto.")
    parser.add_argument("--imgsz",   type=int, default=1280,
                        help="Input image size. 1280 for broadcast, 640 for faster training")
    parser.add_argument("--device",  default="0" if _gpu_available() else "cpu",
                        help="Device: '0' for GPU 0, 'cpu', '0,1' for multi-GPU")
    parser.add_argument("--resume",  type=Path, default=None,
                        help="Resume from checkpoint (.pt file)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without training")
    args = parser.parse_args()

    run_training(args)


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
```

---

## Important: Horizontal Flip and Left/Right Labels

**`fliplr` is set to 0.0 (disabled) by default.** This is intentional.

Horizontal flip augmentation mirrors the image, which means `player_left` becomes
`player_right` and vice versa. If flip augmentation is used without swapping those
labels, the model will be trained on incorrect data.

Two options:
1. **Keep `fliplr=0.0`** (simplest, safe) — no flip augmentation.
2. **Enable flip with label swap**: Create a custom Ultralytics dataset class that
   swaps class 0↔1 and class 2↔3 when a flip is applied. This doubles effective
   training data but requires custom dataloader code.

For the first training run, use option 1 (fliplr=0.0).

---

## Two-Phase Training Strategy

**Phase 1 — Frozen backbone (epochs 1–20):**
The `freeze=10` parameter keeps the first 10 YOLO backbone layers frozen.
Only the detection head learns to output 5 classes. This prevents catastrophic
forgetting of pretrained features during early convergence.

**Phase 2 — Full fine-tune (epochs 21–50):**
After phase 1 converges, resume training with `freeze=0` to fine-tune the
full network end-to-end:

```bash
python scripts/train_yolo.py \
  --resume runs/finetune/soccernet_team_classifier/weights/last.pt \
  --epochs 30
```

In `train_yolo.py`, change `freeze=10` to `freeze=0` for the second run,
or add it as a CLI argument: `--freeze 0`.

---

## GPU Memory Requirements

| Model | Batch | imgsz | VRAM needed |
|---|---|---|---|
| yolo11n | 16 | 640 | ~4 GB |
| yolo11s | 16 | 640 | ~6 GB |
| yolo11s | 16 | 1280 | ~10 GB |
| yolo11m | 16 | 1280 | ~16 GB |

If VRAM is limited:
- Reduce `--batch 8` or `--batch 4`
- Use `--imgsz 640` (faster, slightly less accurate on distant players)
- Use `yolo11n.pt` instead of `yolo11s.pt`

---

## Monitoring Training

YOLO writes training metrics to `runs/finetune/soccernet_team_classifier/results.csv`.
Key metrics to watch:

- `metrics/mAP50` — overall mean AP at 50% IoU. Target: > 0.80 after 50 epochs.
- `metrics/mAP50-95` — stricter metric. Target: > 0.55.
- `train/cls_loss` — should decrease steadily. If it plateaus early, increase `cls` weight.
- Per-class AP in `val/` logs — goalkeeper classes (2, 3) will be lower initially due to
  class imbalance. Acceptable if AP > 0.60 for goalkeepers.

Stop training early if `mAP50` hasn't improved for 10 consecutive epochs (YOLO's
built-in early stopping handles this automatically via `patience=50` default).

---

## Class Imbalance Handling

Goalkeepers (classes 2 and 3) have ~20× fewer samples than outfield players.
YOLO handles this reasonably well via its focal loss, but if goalkeeper mAP is
poor after training:

1. Increase `--sample-rate` during conversion to include more frames (more keeper appearances)
2. Add per-class weight overrides in a custom `train()` call via `cls_pw` parameter
3. As a last resort: oversample clips where goalkeepers are frequently in frame
   (penalty areas, goal kicks) using `--max-clips` with manual clip selection

---

## Expected Training Time

| Hardware | Model | Epochs | Estimated time |
|---|---|---|---|
| RTX 3090 | yolo11s | 50 | ~3–4 hours |
| RTX 4090 | yolo11s | 50 | ~1.5–2 hours |
| T4 (Colab) | yolo11n | 50 | ~5–6 hours |
| CPU only | yolo11n | 10 | ~8–12 hours |
