"""Event-detection validation against StatsBomb open-data ground truth.

Public API
----------
>>> from src.validation.events import RunSpec, validate_run
>>> result = validate_run(RunSpec(
...     name="England_clip",
...     run_dir="data/insights/England_preprocessed_20260625_174216",
...     statsbomb_events="data/statsbomb/events/3795506.json",
...     clip_start_s=0.0,
... ))
>>> result.overall_f1
"""

from __future__ import annotations

from .harness import RunSpec, validate_from_config, validate_run
from .matching import TypeResult, ValidationResult, validate_events
from .model import ALL_TYPES, DIAGNOSTIC_TYPES, SCORED_TYPES, NormEvent
from .ours import load_our_events
from .report import render_markdown, result_to_dict
from .statsbomb import load_statsbomb_events

__all__ = [
    "RunSpec",
    "validate_run",
    "validate_from_config",
    "validate_events",
    "ValidationResult",
    "TypeResult",
    "NormEvent",
    "SCORED_TYPES",
    "DIAGNOSTIC_TYPES",
    "ALL_TYPES",
    "load_our_events",
    "load_statsbomb_events",
    "render_markdown",
    "result_to_dict",
]
