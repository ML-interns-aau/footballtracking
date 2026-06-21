"""Answer user questions grounded strictly in the processed match data."""

from __future__ import annotations

from ai.data_loader import MatchContext
from ai.gemini_client import GeminiClient
from ai.prompt_builder import build_chat_prompt


def answer_question(
    context: MatchContext,
    question: str,
    client: GeminiClient,
    history: list[dict[str, str]] | None = None,
) -> str:
    system, prompt = build_chat_prompt(context.to_prompt_text(), question, history)
    return client.generate(prompt, system_instruction=system, temperature=0.2)
