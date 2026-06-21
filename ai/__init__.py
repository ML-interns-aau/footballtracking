"""Gemini-powered match analysis layer.

Consumes the pipeline's existing outputs (analytics.json, events.json,
player_summary.csv, possession_summary.csv) and turns them into natural-language
match reports and a grounded question-answering assistant.
"""

from ai.data_loader import MatchContext, build_match_context, load_match_data
from ai.gemini_client import GeminiClient, GeminiConfigError, GeminiError, DEFAULT_MODEL
from ai.match_report import generate_match_report
from ai.chat_assistant import answer_question

__all__ = [
    "MatchContext",
    "build_match_context",
    "load_match_data",
    "GeminiClient",
    "GeminiConfigError",
    "GeminiError",
    "DEFAULT_MODEL",
    "generate_match_report",
    "answer_question",
]
