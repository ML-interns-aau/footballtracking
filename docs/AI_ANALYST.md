# AI Analyst — Multi-Provider Match Analysis

The **AI Analyst** page turns a processed match into natural language. It adds three
features on top of the existing pipeline without changing how analysis runs:

- **AI Match Report** — an on-demand tactical report (executive summary, tactical
  analysis, key players, momentum, and commentary).
- **AI Football Assistant** — a chat assistant that answers questions about the
  match, grounded strictly in the processed data.
- **Compare Providers** — runs the same task across every configured provider and
  shows quality, latency, and estimated cost side by side.

The analysis is **provider-agnostic**. The same grounded prompts run against any
configured provider — currently **Google Gemini** (`google-genai`) and **Groq**
(`groq`). Pick one from the selector at the top of the page, or compare them.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

This installs `google-genai`, `groq`, and `python-dotenv` alongside the existing packages.

### 2. Provide at least one API key

Configure any provider you want to use. The dashboard automatically offers the ones
that have a key set, and flags the ones that don't.

```bash
# .env  (copy from .env.example — gitignored)
GEMINI_API_KEY=your_key_here    # https://aistudio.google.com/apikey
GROQ_API_KEY=your_key_here      # https://console.groq.com/keys
```

Keys are read from the environment only; they are never stored in source. You only
need one to use the report and assistant — set two (or more) to use the comparison.

### 3. Run the app

```bash
streamlit run app/Home.py
```

Open the **AI Analyst** tab, pick a processed game, choose a **provider** and
**model**, then use **AI Match Report**, **AI Football Assistant**, or **Compare
Providers**.

---

## Switching providers

The provider and model selectors sit at the top of the page:

- **Provider** — every registered provider is listed; those without an API key are
  marked `• no API key` and show inline setup help when selected.
- **Model** — the models available for the selected provider (e.g.
  `gemini-2.5-flash`, `llama-3.3-70b-versatile`).

Switching the provider re-runs the *same* prompts against the new model. Reports are
cached per `(game, provider, model)`, so you can generate with Gemini, switch to
Groq, and still have both results when you switch back.

Every report and chat answer carries a one-line footer: **provider · model · latency
· tokens · estimated cost**.

---

## Comparing quality, latency, and cost

The **Compare Providers** tab runs one task — the full match report or a custom
question — against every configured provider (each at its default model) and renders:

- a **summary table**: latency, token usage, estimated cost, and output length;
- the **fastest** and **cheapest** provider for that run;
- each provider's **full output** in its own sub-tab, so you can judge quality directly.

> **Cost figures are estimates.** They are computed from published list prices (see
> `_PRICING` in `ai/llm_provider.py`, indicative as of early 2026) times the token
> counts reported by each API. They ignore free tiers and discounts and will drift —
> update the table as prices change.

---

## How grounding works

The AI layer never re-analyzes video. It consumes the artifacts a completed run
already writes to `data/insights/<game_id>/`:

| File | Used for |
|---|---|
| `analytics.json` | team names, per-frame possession → momentum segments |
| `events.json` | passes, recoveries, interceptions, crosses, zone entries → timelines and attacking/defensive counts |
| `player_summary.csv` | per-player distance, speed, possession → top performers |
| `possession_summary.csv` | team possession percentages |

`ai/data_loader.py` compiles these into a compact `MatchContext`, rendered to a
plain-text block and injected into every prompt. The system prompts instruct the
model to answer **only** from that block, quote exact figures, and never assume goals
or scorelines that tracking data does not contain. When data is missing, the
assistant replies: *"I couldn't find that information in the processed match data."*

---

## Module layout

```
ai/
├── llm_provider.py     # provider-agnostic base: LLMProvider, LLMResponse, pricing, registry/factory
├── gemini_client.py    # Gemini provider (google-genai): keys, model, usage, error translation
├── groq_client.py      # Groq provider (groq SDK): same contract
├── data_loader.py      # load existing outputs → grounded MatchContext
├── prompt_builder.py   # report + chat prompts (grounding rules live here)
├── match_report.py     # generate the five-section report (any provider)
└── chat_assistant.py   # answer a question from the context (any provider)
```

`create_provider(name, model)` is the single entry point for getting a client.
`generate()` returns an `LLMResponse` carrying the text plus latency, token usage,
and estimated cost — that uniform return is what powers provider switching and the
comparison tab.

### Adding another provider

1. Subclass `LLMProvider` and implement `_complete(...) -> (text, input_tokens, output_tokens)`.
2. Register a `ProviderSpec` in `_SPECS` (name, label, env key, default + available models).
3. Wire it into `create_provider`. Add its prices to `_PRICING` for cost estimates.

Nothing else in the app changes — the report, assistant, and comparison pick it up automatically.

Streamlit integration lives in `app/pages/ai_analyst_page.py` and is wired into the
existing navbar (`app/utils.py` → `NAV_PAGES`, routed from `app/Home.py`).

---

## Performance & caching

- Each `(provider, model)` client is built once via `st.cache_resource` and reused.
- Match context is parsed once per game and cached via `st.cache_data`, keyed on the
  output files' modification time, so re-processing a game invalidates the cache.
- A generated report is held in session state per `(game, provider, model)`.

## Error handling

The page degrades gracefully and never crashes the app:

- Missing key / SDK → an inline setup message for that provider; buttons disable.
- Missing or empty match data → a prompt to run the pipeline first.
- API failures, invalid key, rate limits, network errors → a user-friendly message.
- In the comparison, a provider that fails is reported in its row/sub-tab while the
  others still complete.
