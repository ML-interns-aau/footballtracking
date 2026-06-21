"""Generate the natural-language match report from a grounded context."""

from __future__ import annotations

from ai.data_loader import MatchContext
from ai.gemini_client import GeminiClient
from ai.prompt_builder import build_report_prompt


def generate_match_report(context: MatchContext, client: GeminiClient) -> str:
    system, prompt = build_report_prompt(context.to_prompt_text())
    return client.generate(prompt, system_instruction=system, temperature=0.5)
