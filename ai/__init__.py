"""Multi-provider LLM match-analysis layer.

Consumes the pipeline's existing outputs (analytics.json, events.json,
player_summary.csv, possession_summary.csv) and turns them into natural-language
match reports and a grounded question-answering assistant.

The analysis is provider-agnostic: pick a provider (``"gemini"``, ``"groq"``,
...) via :func:`ai.llm_provider.create_provider` and the same report/chat code
runs against it, returning an :class:`~ai.llm_provider.LLMResponse` carrying the
text plus latency, token usage, and estimated cost for comparison.
"""

from ai.chat_assistant import answer_question
from ai.data_loader import MatchContext, build_match_context, load_match_data
from ai.gemini_client import DEFAULT_MODEL, GeminiClient, GeminiConfigError, GeminiError
from ai.groq_client import GroqClient, GroqConfigError, GroqError
from ai.llm_provider import (
    LLMConfigError,
    LLMError,
    LLMProvider,
    LLMResponse,
    ProviderSpec,
    configured_providers,
    create_provider,
    estimate_cost,
    get_spec,
    list_specs,
    provider_configured,
)
from ai.match_report import generate_match_report

__all__ = [
    # Data
    "MatchContext",
    "build_match_context",
    "load_match_data",
    # Provider abstraction
    "LLMProvider",
    "LLMResponse",
    "LLMConfigError",
    "LLMError",
    "ProviderSpec",
    "create_provider",
    "list_specs",
    "get_spec",
    "configured_providers",
    "provider_configured",
    "estimate_cost",
    # Concrete providers
    "GeminiClient",
    "GeminiConfigError",
    "GeminiError",
    "GroqClient",
    "GroqConfigError",
    "GroqError",
    "DEFAULT_MODEL",
    # Tasks
    "generate_match_report",
    "answer_question",
]
