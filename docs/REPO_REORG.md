# Repository Reorganisation — Migration Guide

This document records every file move made during the June 2026 repo-structure cleanup.
No runtime logic, public function signatures, or API surfaces were changed.

---

## What changed and why

| Category | Before | After | Reason |
|---|---|---|---|
| **Docs** | `HOW_TO_USE_GUIDES.md` | `docs/HOW_TO_USE_GUIDES.md` | Consolidate docs under `docs/` |
| **Docs** | `IMPLEMENTATION_ROADMAP.md` | `docs/IMPLEMENTATION_ROADMAP.md` | " |
| **Docs** | `PRODUCTION_MASTERCLASS.md` | `docs/PRODUCTION_MASTERCLASS.md` | " |
| **Docs** | `QUICK_REFERENCE.md` | `docs/QUICK_REFERENCE.md` | " |
| **Docs** | `CONFIGURATION_GUIDE.md` | `docs/CONFIGURATION_GUIDE.md` | " |
| **Docs** | `DOCKER.md` | `docs/DOCKER.md` | " |
| **Docs** | `implementation_plan.md` | `docs/implementation_plan.md` | " |
| **Setup scripts** | `download_model.py` | `scripts/download_model.py` | One-time env setup; belongs with `scripts/` |
| **Setup scripts** | `install_deps.py` | `scripts/install_deps.py` | " |
| **Dev tools** | `generate_test_data.py` | `tools/generate_test_data.py` | Dev/test utility; belongs with `tools/` |
| **Dev tools** | `generate_and_export_200f.py` | `tools/generate_and_export_200f.py` | " |
| **Dev tools** | `post_process_results.py` | `tools/post_process_results.py` | " |

Files that stayed at root: `README.md`, `QUICKSTART.md`, `main.py`, `requirements.txt`,
`Makefile`, `Dockerfile`, `docker-compose*.yml`, `start.ps1`, `run_app.bat`, `install_remaining.bat`.

---

## Unchanged public API

The following modules were noted as off-limits and were **not touched**:

- `src/pipeline/data_exporter.py` — `DataExporter` class, all methods
- `src/pipeline/events.py` — `EventsDetector` class, all methods
- `app/pages/preprocess_page.py` — Streamlit UI logic
- `app/pages/analysis_page.py` — Streamlit UI logic
- `main.py` — CLI entry point and `main()` signature

---

## Legacy / dead-code modules (not deleted, flagged for future cleanup)

These packages exist under `src/` but are **not imported anywhere** in the active codebase.
They predate the current `src/pipeline/` architecture and can be removed in a future PR
once confirmed no other consumers exist.

| Package | Files |
|---|---|
| `src/detection_tracking/` | `__init__.py`, `detect_ball.py`, `detect_players.py`, `tracker.py` |
| `src/output/` | `export_csv.py`, `export_json.py` |

---

## Import path convention for scripts in `tools/` and `scripts/`

All scripts that import from `src/` must bootstrap the project root onto `sys.path`
**before** any `src.*` import:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.foo import Bar  # works from tools/ or scripts/
```

This pattern is already used by `tools/annotate_from_frames.py` and
`tools/generate_and_export_200f.py`.

---

## .gitignore changes

| Change | Detail |
|---|---|
| Bug fix | Removed accidental `/.gitignore` self-exclusion line |
| Added | `venv311/` (second local venv was untracked) |
| Added | `*.egg-info/`, `dist/`, `build/`, `.eggs/` (Python packaging artifacts) |
| Added | `.idea/`, `*.swp`, `*.swo`, `desktop.ini` (more editor/OS noise) |
| Added | `*.pkl`, `*.h5` (additional ML artefact extensions) |
| Added | `.coverage`, `htmlcov/`, `.tox/` (test coverage artefacts) |
| Added | `.env`, `.env.*` with `!.env.example` carve-out (secrets safety) |
| Added | `*.ipynb` (notebook files) |
| Reorganised | Sections grouped with comment headers for readability |
