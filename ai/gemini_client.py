"""Google Gemini provider (google-genai SDK).

Implements the :class:`~ai.llm_provider.LLMProvider` contract: key resolution,
model selection, token-usage reporting, and error translation, so the rest of
the application never touches the SDK directly.
"""

from __future__ import annotations

from ai.llm_provider import (
    LLMConfigError,
    LLMError,
    LLMProvider,
    resolve_api_key,
)

DEFAULT_MODEL = "gemini-1.5-flash"
API_KEY_ENV = "GEMINI_API_KEY"


# Backwards-compatible aliases. Existing call sites catch these names; keeping
# them as subclasses of the shared errors means generic ``except LLMError``
# handlers also catch Gemini failures.
class GeminiConfigError(LLMConfigError):
    """Client could not be configured — missing key or missing dependency."""


class GeminiError(LLMError):
    """A Gemini API call failed at request time."""


def get_api_key() -> str:
    try:
        return resolve_api_key(API_KEY_ENV)
    except LLMConfigError as exc:
        raise GeminiConfigError(str(exc)) from exc


def _describe_api_error(exc: Exception) -> str:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    message = getattr(exc, "message", None) or str(exc)
    upper = str(message).upper()
    if code in (401, 403) or "API_KEY" in upper or "PERMISSION" in upper:
        return "Gemini rejected the request — verify GEMINI_API_KEY is valid and authorized."
    if code == 429 or "RATE" in upper or "QUOTA" in upper:
        return "Gemini rate limit or quota reached. Wait a moment and try again."
    try:
        if code is not None and 500 <= int(code) < 600:
            return "Gemini is temporarily unavailable. Please try again shortly."
    except (TypeError, ValueError):
        pass
    return f"Gemini API error: {message}"


def _usage_counts(response) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None
    in_tok = getattr(usage, "prompt_token_count", None)
    out_tok = getattr(usage, "candidates_token_count", None)
    return in_tok, out_tok


class GeminiClient(LLMProvider):
    """Reusable Gemini client. Construct once and share across requests."""

    name = "gemini"
    label = "Google Gemini"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        super().__init__(model)
        try:
            from google import genai
        except ImportError as exc:
            raise GeminiConfigError(
                "The google-genai package is not installed. Run: pip install google-genai"
            ) from exc
        try:
            self._client = genai.Client(api_key=api_key or get_api_key())
        except GeminiConfigError:
            raise
        except Exception as exc:
            raise GeminiConfigError(f"Failed to initialize Gemini client: {exc}") from exc

    def _complete(
        self, prompt: str, system_instruction: str | None, temperature: float
    ) -> tuple[str, int | None, int | None]:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            raise GeminiError(_describe_api_error(exc)) from exc

        text = getattr(response, "text", None) or ""
        in_tok, out_tok = _usage_counts(response)
        return text, in_tok, out_tok
