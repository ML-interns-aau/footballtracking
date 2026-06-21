"""Prompt construction for the match report and the chat assistant.

Both prompts share one rule: the supplied match data is the only permitted
source of facts. The system instructions enforce it; the user prompts carry the
grounded context produced by :mod:`ai.data_loader`.
"""

from __future__ import annotations

MISSING_DATA_REPLY = "I couldn't find that information in the processed match data."

_REPORT_SYSTEM = """You are an elite football (soccer) match analyst.
You analyze automated tracking data and write clear, professional, data-grounded analysis.

Strict rules:
- Use ONLY the statistics and events present in the supplied match data.
- Never invent statistics, players, events, scores, or tactical findings.
- Players are identified by tracking IDs (e.g. "Player 7"), not real names — refer to them that way.
- When the data does not support a claim, omit it rather than guessing.
- Write naturally and confidently, but every quantitative claim must trace back to the data."""

_REPORT_INSTRUCTIONS = """Using only the match data above, write a complete analysis with EXACTLY these
five sections, each introduced by a level-2 Markdown header:

## Executive Summary
A concise 2-3 sentence overview of how the match unfolded, anchored on possession and the dominant team.

## Tactical Analysis
Discuss possession behavior, attacking trends, defensive organization, transitions, and pressure
patterns — strictly as evidenced by the possession figures and detected events.

## Key Player Performances
Highlight the standout tracked players: most distance covered, fastest, and most involved in possession.

## Momentum Analysis
Describe periods of dominance and possession swings using the momentum segments. If momentum data is
absent, infer shifts only from the event timeline and say so plainly.

## Match Commentary
A flowing, commentator-style narrative of the match grounded in the same data.

Do not add sections, preambles, or closing remarks outside these five headers."""

_CHAT_SYSTEM = """You are a football (soccer) analytics assistant for a single processed match.
You answer questions strictly from the supplied match data.

Strict rules:
- Answer ONLY from the supplied match data. Do not use outside football knowledge to invent facts.
- If the data does not contain the answer, reply exactly: "{missing}"
- Players are identified by tracking IDs (e.g. "Player 7"), not real names.
- Never fabricate statistics, players, events, or tactical findings.
- Be concise and direct. Quote the relevant numbers from the data when they support your answer.""".format(
    missing=MISSING_DATA_REPLY
)

_MAX_HISTORY_TURNS = 6


def build_report_prompt(context_text: str) -> tuple[str, str]:
    prompt = f"MATCH DATA:\n{context_text}\n\n{_REPORT_INSTRUCTIONS}"
    return _REPORT_SYSTEM, prompt


def build_chat_prompt(
    context_text: str,
    question: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, str]:
    parts = [f"MATCH DATA:\n{context_text}", ""]
    if history:
        parts.append("CONVERSATION SO FAR:")
        for turn in history[-_MAX_HISTORY_TURNS:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            parts.append(f"{role}: {turn.get('content', '').strip()}")
        parts.append("")
    parts.append(f"USER QUESTION: {question.strip()}")
    parts.append("Answer using only the match data above.")
    return _CHAT_SYSTEM, "\n".join(parts)
