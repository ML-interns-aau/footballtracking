# Copilot Instructions — Football Team Classifier: Full Improvement Pipeline

## Project Overview

This project is a football (soccer) video analysis system. The core problem being solved
is accurate team classification — assigning each detected player to their team, and
distinguishing goalkeepers and referees from outfield players.

The current implementation (`TeamClassifier` in `team_classifier.py`) uses raw HSV
color + KMeans clustering. It works but is brittle to lighting changes, similar kit
colors, and green kits.

The goal of this multi-phase pipeline is:
1. Improve the existing color-based classifier immediately (no training required)
2. Download and parse the SoccerNet-GSR dataset properly
3. Convert the dataset annotations to YOLO format
4. Fine-tune a YOLO model on those annotations to predict 5 classes natively:
   `player_left`, `player_right`, `goalkeeper_left`, `goalkeeper_right`, `referee`

---

## Instruction Files in This Folder

Each file covers one phase. They are designed to be used sequentially but are
self-contained so Copilot can be focused on one phase at a time.

| File | Phase | When to Use |
|---|---|---|
| `01-color-classifier-improvements.md` | Immediate color pipeline fixes | Start here — no data needed |
| `02-dataset-download-and-structure.md` | SoccerNet-GSR acquisition | After phase 1 is working |
| `03-annotation-conversion.md` | JSON → YOLO label conversion | After dataset is downloaded |
| `04-yolo-finetune.md` | Fine-tuning YOLO on the converted data | After conversion is complete |
| `05-integration.md` | Wiring the fine-tuned model back into the pipeline | After training converges |

---

## Class Map (Used Throughout All Files)

This is the canonical class mapping for the YOLO fine-tune target. Use it consistently
across all conversion, training, and integration code.

```python
CLASS_MAP = {
    "player_left":       0,   # outfield player, team on the left side of pitch
    "player_right":      1,   # outfield player, team on the right side of pitch
    "goalkeeper_left":   2,   # goalkeeper, left team
    "goalkeeper_right":  3,   # goalkeeper, right team
    "referee":           4,   # any referee role (main, assistant)
}
```

The SoccerNet-GSR annotation uses `role` ∈ {`player`, `goalkeeper`, `referee`, `other`}
and `team` ∈ {`left`, `right`}. The mapping from those to CLASS_MAP is:

```
role=player,     team=left    → 0  (player_left)
role=player,     team=right   → 1  (player_right)
role=goalkeeper, team=left    → 2  (goalkeeper_left)
role=goalkeeper, team=right   → 3  (goalkeeper_right)
role=referee,    team=any     → 4  (referee)
role=other,      team=any     → SKIP (do not include in YOLO labels)
```

---

## Key Paths and Conventions

```
project_root/
├── team_classifier.py          ← phase 1 target
├── detector.py                 ← YOLO inference wrapper (phase 5 target)
├── data/
│   └── SoccerNetGS/
│       ├── train/              ← downloaded dataset splits
│       ├── valid/
│       └── test/
├── datasets/
│   └── soccernet_yolo/         ← converted YOLO dataset (phase 3 output)
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       ├── labels/
│       │   ├── train/
│       │   └── val/
│       └── dataset.yaml        ← YOLO training config
├── runs/
│   └── finetune/               ← YOLO training output (phase 4)
└── scripts/
    ├── download_dataset.py     ← phase 2
    ├── convert_annotations.py  ← phase 3
    └── train_yolo.py           ← phase 4
```

---

## Hard Rules (Apply to All Phases)

- Never hardcode absolute paths. Use `pathlib.Path` everywhere.
- All scripts must be runnable from the project root: `python scripts/script_name.py`
- All scripts must have a `--help` flag via `argparse`.
- Log progress with `tqdm` for any loop over files or frames.
- Do not silently skip errors. Use `try/except` with logging, not bare `pass`.
- Python 3.9+ syntax only (match statements, `|` union types are fine).
- GPU is optional everywhere — all code must fall back to CPU gracefully.
