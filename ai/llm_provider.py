"""Provider-agnostic LLM layer.

The rest of the application never talks to a vendor SDK directly. It asks for a
provider by name (``"gemini"``, ``"groq"``, ...) and gets back an
:class:`LLMProvider` whose :meth:`~LLMProvider.generate` returns a rich
:class:`LLMResponse` — the text plus the latency, token usage, and an estimated
cost. That uniform response is what makes provider switching and side-by-side
comparison in the dashboard possible.

Adding a provider means: write a subclass implementing ``_complete`` and register
it in :data:`_SPECS`. Nothing else in the app needs to change.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

# ── Errors ──────────────────────────────────────────────────────────────────


class LLMConfigError(RuntimeError):
    """Provider could not be configured — missing key or missing dependency."""


class LLMError(RuntimeError):
    """A provider API call failed at request time."""


# ── Response ────────────────────────────────────────────────────────────────


@dataclass
class LLMResponse:
    """One completion plus the metadata needed to compare providers."""

    text: str
    provider: str
    label: str
    model: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


# ── Pricing ─────────────────────────────────────────────────────────────────
#
# USD per 1,000,000 tokens as (input_rate, output_rate). These are published
# list prices and are indicative only (collected early 2026) — they drift, and
# free tiers/discounts are not reflected. Cost figures in the UI are estimates.
_PRICING: dict[str, tuple[float, float]] = {
    # Google Gemini
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    # Groq (open models, served fast)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-oss-20b": (0.10, 0.50),
}


def estimate_cost(
    model: str, input_tokens: int | None, output_tokens: int | None
) -> float | None:
    """Estimated USD cost for a call, or ``None`` if it can't be computed."""
    price = _PRICING.get(model)
    if price is None or input_tokens is None or output_tokens is None:
        return None
    in_rate, out_rate = price
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


# ── Shared key handling ───────────────────────────────────────────────────────

_dotenv_loaded = False


def _load_dotenv_once() -> None:
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def resolve_api_key(env_key: str, override: str | None = None) -> str:
    if override:
        return override.strip()
    _load_dotenv_once()
    key = os.environ.get(env_key, "").strip()
    if not key:
        raise LLMConfigError(
            f"{env_key} is not set. Add it to your environment or a .env file."
        )
    return key


def has_api_key(env_key: str) -> bool:
    _load_dotenv_once()
    return bool(os.environ.get(env_key, "").strip())


# ── Base provider ─────────────────────────────────────────────────────────────


class LLMProvider(ABC):
    """A configured client for one model. Construct once and reuse."""

    #: stable id used in the registry (e.g. "gemini")
    name: str = ""
    #: human-readable name shown in the UI (e.g. "Google Gemini")
    label: str = ""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def _complete(
        self, prompt: str, system_instruction: str | None, temperature: float
    ) -> tuple[str, int | None, int | None]:
        """Return ``(text, input_tokens, output_tokens)``. Token counts may be
        ``None`` when the provider does not report usage."""

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.4,
    ) -> LLMResponse:
        start = time.perf_counter()
        text, in_tok, out_tok = self._complete(prompt, system_instruction, temperature)
        latency = time.perf_counter() - start
        text = (text or "").strip()
        if not text:
            raise LLMError(f"{self.label or self.name} returned an empty response.")
        return LLMResponse(
            text=text,
            provider=self.name,
            label=self.label or self.name,
            model=self.model,
            latency_s=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=estimate_cost(self.model, in_tok, out_tok),
        )


# ── Registry ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    label: str
    env_key: str
    default_model: str
    models: tuple[str, ...]


_SPECS: dict[str, ProviderSpec] = {
    "gemini": ProviderSpec(
        name="gemini",
        label="Google Gemini",
        env_key="GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
        models=("gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"),
    ),
    "groq": ProviderSpec(
        name="groq",
        label="Groq",
        env_key="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        models=("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b"),
    ),
}


def list_specs() -> list[ProviderSpec]:
    return list(_SPECS.values())


def get_spec(name: str) -> ProviderSpec:
    try:
        return _SPECS[name]
    except KeyError:
        raise LLMConfigError(f"Unknown LLM provider: {name!r}") from None


def provider_configured(name: str) -> bool:
    """True when an API key for this provider is present in the environment."""
    return has_api_key(get_spec(name).env_key)


def configured_providers() -> list[ProviderSpec]:
    return [s for s in _SPECS.values() if has_api_key(s.env_key)]


def create_provider(name: str, model: str | None = None) -> LLMProvider:
    """Instantiate a provider by name. Raises :class:`LLMConfigError` when the
    key or SDK is missing (caught and surfaced nicely by the dashboard)."""
    spec = get_spec(name)
    model = model or spec.default_model
    if name == "gemini":
        from ai.gemini_client import GeminiClient

        return GeminiClient(model=model)
    if name == "groq":
        from ai.groq_client import GroqClient

        return GroqClient(model=model)
    raise LLMConfigError(f"No client implemented for provider {name!r}.")
