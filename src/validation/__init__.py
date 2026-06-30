"""Validation subsystem for the football analytics pipeline.

Currently exposes the :mod:`src.validation.events` package, a harness that
scores the pipeline's detected events (``events.json``) against StatsBomb
open-data event data and reports precision / recall / F1 per event type.
"""
