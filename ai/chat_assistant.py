"""Answer user questions grounded strictly in the processed match data."""

from __future__ import annotations

from ai.data_loader import MatchContext
from ai.llm_provider import LLMProvider, LLMResponse
from ai.prompt_builder import build_chat_prompt


def answer_question(
    context: MatchContext,
    question: str,
    client: LLMProvider,
    history: list[dict[str, str]] | None = None,
) -> LLMResponse:
    """Answer one question from the match data using any configured provider."""
    system, prompt = build_chat_prompt(context.to_prompt_text(), question, history)
    return client.generate(prompt, system_instruction=system, temperature=0.2)
