"""Groq provider (groq SDK).

Groq serves open models (Llama 3.x, gpt-oss, ...) on its LPU hardware with very
low latency, which makes it a useful counterpoint to Gemini in the dashboard's
provider comparison. The chat-completions API is OpenAI-shaped.
"""

from __future__ import annotations

from ai.llm_provider import (
    LLMConfigError,
    LLMError,
    LLMProvider,
    resolve_api_key,
)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
API_KEY_ENV = "GROQ_API_KEY"


class GroqConfigError(LLMConfigError):
    """Client could not be configured — missing key or missing dependency."""


class GroqError(LLMError):
    """A Groq API call failed at request time."""


def get_api_key() -> str:
    try:
        return resolve_api_key(API_KEY_ENV)
    except LLMConfigError as exc:
        raise GroqConfigError(str(exc)) from exc


def _describe_api_error(exc: Exception) -> str:
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    message = getattr(exc, "message", None) or str(exc)
    upper = str(message).upper()
    if code in (401, 403) or "API_KEY" in upper or "INVALID_API_KEY" in upper:
        return "Groq rejected the request — verify GROQ_API_KEY is valid and authorized."
    if code == 429 or "RATE" in upper or "QUOTA" in upper:
        return "Groq rate limit or quota reached. Wait a moment and try again."
    if code == 404 or ("MODEL" in upper and "NOT" in upper):
        return "Groq could not find the requested model. Pick a different model."
    try:
        if code is not None and 500 <= int(code) < 600:
            return "Groq is temporarily unavailable. Please try again shortly."
    except (TypeError, ValueError):
        pass
    return f"Groq API error: {message}"


def _usage_counts(response) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    return getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)


class GroqClient(LLMProvider):
    """Reusable Groq client. Construct once and share across requests."""

    name = "groq"
    label = "Groq"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        super().__init__(model)
        try:
            from groq import Groq
        except ImportError as exc:
            raise GroqConfigError(
                "The groq package is not installed. Run: pip install groq"
            ) from exc
        try:
            self._client = Groq(api_key=api_key or get_api_key())
        except GroqConfigError:
            raise
        except Exception as exc:
            raise GroqConfigError(f"Failed to initialize Groq client: {exc}") from exc

    def _complete(
        self, prompt: str, system_instruction: str | None, temperature: float
    ) -> tuple[str, int | None, int | None]:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
        except Exception as exc:
            raise GroqError(_describe_api_error(exc)) from exc

        text = ""
        if response.choices:
            text = response.choices[0].message.content or ""
        in_tok, out_tok = _usage_counts(response)
        return text, in_tok, out_tok
