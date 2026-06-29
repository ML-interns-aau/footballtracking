"""Generate the natural-language match report from a grounded context."""

from __future__ import annotations

from ai.data_loader import MatchContext
from ai.llm_provider import LLMProvider, LLMResponse
from ai.prompt_builder import build_report_prompt


def generate_match_report(context: MatchContext, client: LLMProvider) -> LLMResponse:
    """Produce the five-section match report using any configured provider."""
    system, prompt = build_report_prompt(context.to_prompt_text())
    return client.generate(prompt, system_instruction=system, temperature=0.5)
