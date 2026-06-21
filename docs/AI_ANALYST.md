# AI Analyst — Gemini-Powered Match Analysis

The **AI Analyst** page turns a processed match into natural language. It adds two
features on top of the existing pipeline without changing how analysis runs:

- **AI Match Report** — an on-demand tactical report (executive summary, tactical
  analysis, key players, momentum, and commentary).
- **AI Football Assistant** — a chat assistant that answers questions about the
  match, grounded strictly in the processed data.

Everything is powered exclusively by Google **Gemini** (`gemini-2.5-flash`) via the
`google-genai` SDK.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

This installs `google-genai` and `python-dotenv` alongside the existing packages.

### 2. Provide a Gemini API key

Get a key from [Google AI Studio](https://aistudio.google.com/apikey), then either
export it or drop it in a `.env` file at the project root:

```bash
# .env  (copy from .env.example — gitignored)
GEMINI_API_KEY=your_key_here
```

The key is read from the environment only; it is never stored in source.

### 3. Run the app

```bash
streamlit run app/Home.py
```

Open the **AI Analyst** tab in the top navigation. Pick a processed game, then use
the **AI Match Report** or **AI Football Assistant** tab.

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

`ai/data_loader.py` compiles these into a compact `MatchContext`, which is rendered
to a plain-text block and injected into every prompt. The system prompts instruct
Gemini to answer **only** from that block. When data is missing, the assistant
replies: *"I couldn't find that information in the processed match data."*

---

## Module layout

```
ai/
├── gemini_client.py    # google-genai wrapper: key handling, model, error translation
├── data_loader.py      # load existing outputs → grounded MatchContext
├── prompt_builder.py   # report + chat prompts (grounding rules live here)
├── match_report.py     # generate the five-section report
└── chat_assistant.py   # answer a question from the context
```

Streamlit integration lives in `app/pages/ai_analyst_page.py` and is wired into the
existing navbar (`app/utils.py` → `NAV_PAGES`, routed from `app/Home.py`).

---

## Performance & caching

- The Gemini client is built once via `st.cache_resource` and reused.
- Match context is parsed once per game and cached via `st.cache_data`, keyed on the
  output files' modification time, so re-processing a game invalidates the cache.
- A generated report is held in session state per game; regenerate explicitly.

## Error handling

The page degrades gracefully and never crashes the app:

- Missing key / SDK → an inline setup message.
- Missing or empty match data → a prompt to run the pipeline first.
- API failures, invalid key, rate limits, network errors → a user-friendly message.
