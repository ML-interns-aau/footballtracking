"""Thin wrapper around the google-genai SDK.

Centralises API-key resolution, model selection, and error translation so the
rest of the application never touches the SDK directly.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "gemini-2.5-flash"
API_KEY_ENV = "GEMINI_API_KEY"


class GeminiConfigError(RuntimeError):
    """Client could not be configured — missing key or missing dependency."""


class GeminiError(RuntimeError):
    """A Gemini API call failed at request time."""


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


def get_api_key() -> str:
    _load_dotenv_once()
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise GeminiConfigError(
            f"{API_KEY_ENV} is not set. Add it to your environment or a .env file."
        )
    return key


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


class GeminiClient:
    """Reusable Gemini client. Construct once and share across requests."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        try:
            from google import genai
        except ImportError as exc:
            raise GeminiConfigError(
                "The google-genai package is not installed. Run: pip install google-genai"
            ) from exc
        self.model = model
        try:
            self._client = genai.Client(api_key=api_key or get_api_key())
        except GeminiConfigError:
            raise
        except Exception as exc:
            raise GeminiConfigError(f"Failed to initialize Gemini client: {exc}") from exc

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.4,
    ) -> str:
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

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise GeminiError("Gemini returned an empty response.")
        return text
