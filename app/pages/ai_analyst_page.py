# app/pages/ai_analyst_page.py
"""AI Analyst Page — multi-provider match report and grounded chat assistant.

Consumes the existing pipeline outputs for a selected game and exposes three
tabs: an on-demand natural-language match report, a question-answering
assistant, and a side-by-side provider comparison (quality, latency, cost).

The analysis is provider-agnostic — pick Gemini or Groq from the selector at the
top of the page and the same grounded prompts run against it. Every answer is
built exclusively from the processed match data.
"""

import os

import pandas as pd
import streamlit as st

from app.config import INSIGHTS_DIR, get_game_list
from app.utils import page_header, nav_button, TEXT_MUTED
from src.exporters.output_schema import OutputFiles

from ai.chat_assistant import answer_question
from ai.data_loader import build_match_context
from ai.llm_provider import (
    LLMConfigError,
    LLMError,
    create_provider,
    get_spec,
    list_specs,
    provider_configured,
)
from ai.match_report import generate_match_report

EVENTS_JSON = "events.json"

SUGGESTED_QUESTIONS = [
    "Which team controlled possession?",
    "Who covered the most distance?",
    "What was the fastest sprint?",
    "Which team dominated attacks?",
    "Summarize the match.",
]


@st.cache_resource(show_spinner=False)
def _get_client(provider: str, model: str):
    """One reusable client per (provider, model). Cached across reruns."""
    return create_provider(provider, model)


def _context_signature(game_dir: str) -> float:
    latest = 0.0
    for name in (OutputFiles.ANALYTICS_JSON, EVENTS_JSON,
                 OutputFiles.PLAYER_SUMMARY, OutputFiles.POSSESSION_SUMMARY):
        path = os.path.join(game_dir, name)
        if os.path.exists(path):
            latest = max(latest, os.path.getmtime(path))
    return latest


@st.cache_data(show_spinner=False)
def _load_context(game_dir: str, game_id: str, _signature: float):
    return build_match_context(game_dir, game_id)


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_cost(cost) -> str:
    if cost is None:
        return "n/a"
    if cost < 0.01:
        return f"${cost * 100:.3f}¢"
    return f"${cost:.4f}"


def _fmt_tokens(resp) -> str:
    total = resp.total_tokens
    if total is None:
        return "n/a"
    if resp.input_tokens is not None and resp.output_tokens is not None:
        return f"{total:,} ({resp.input_tokens:,}→{resp.output_tokens:,})"
    return f"{total:,}"


def _render_meta(resp):
    """A muted one-line summary of provider, latency, tokens, and cost."""
    line = (
        f"{resp.label} · {resp.model} · "
        f"{resp.latency_s:.1f}s · {_fmt_tokens(resp)} tok · {_fmt_cost(resp.cost_usd)} est."
    )
    st.markdown(
        f"<div style='font-size:0.72rem;color:{TEXT_MUTED};margin-top:0.35rem;'>{line}</div>",
        unsafe_allow_html=True,
    )


# ── Provider selection ────────────────────────────────────────────────────────

def _provider_controls():
    """Render the provider + model selectors. Returns (provider, model, configured)."""
    specs = list_specs()
    names = [s.name for s in specs]

    # Default to the first configured provider, else the first listed.
    default_idx = 0
    for i, s in enumerate(specs):
        if provider_configured(s.name):
            default_idx = i
            break

    def _label(name: str) -> str:
        s = get_spec(name)
        return s.label if provider_configured(name) else f"{s.label}  •  no API key"

    c1, c2 = st.columns([1, 1])
    with c1:
        provider = st.selectbox(
            "LLM provider",
            options=names,
            index=default_idx,
            format_func=_label,
            key="ai_provider",
            help="Switch the model powering the report and assistant. Both use the same grounded prompts.",
        )
    spec = get_spec(provider)
    with c2:
        model = st.selectbox(
            "Model",
            options=list(spec.models),
            index=0,
            key=f"ai_model_{provider}",
        )

    configured = provider_configured(provider)
    if not configured:
        _render_api_key_help(spec)
    return provider, model, configured


def _render_api_key_help(spec):
    st.warning(f"{spec.label} is not configured.")
    st.markdown(
        f"""
        <div style="font-size:0.85rem; color:{TEXT_MUTED}; line-height:1.6;">
        Set a <code>{spec.env_key}</code> environment variable (or add it to a
        <code>.env</code> file in the project root), then reload this page:
        <pre style="margin-top:0.5rem;">{spec.env_key}=your_key_here</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _resolve_client(provider: str, model: str):
    """Build a client, surfacing config problems inline. Returns the client or None."""
    try:
        return _get_client(provider, model)
    except LLMConfigError as exc:
        st.warning(str(exc))
        return None


# ── Report tab ────────────────────────────────────────────────────────────────

def _render_report_tab(context, game_id: str, provider: str, model: str, configured: bool):
    if not context.has_data:
        st.info("No processed match data found for this game. Run the analysis pipeline first.")
        return

    # Cache the report per (game, provider, model) so switching providers keeps each result.
    report_key = f"ai_report_{game_id}_{provider}_{model}"
    existing = st.session_state.get(report_key)

    c1, c2 = st.columns([1, 1])
    with c1:
        generate = st.button("Generate AI Match Report", type="primary",
                             width='stretch', disabled=not configured,
                             key=f"gen_report_{game_id}")
    with c2:
        if existing and st.button("Regenerate", width='stretch', key=f"regen_report_{game_id}"):
            st.session_state.pop(report_key, None)
            generate = True

    if generate and configured:
        client = _resolve_client(provider, model)
        if client is None:
            return
        try:
            with st.spinner(f"Analyzing match data with {get_spec(provider).label}..."):
                st.session_state[report_key] = generate_match_report(context, client)
        except LLMError as exc:
            st.error(str(exc))
            return

    resp = st.session_state.get(report_key)
    if resp:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown(resp.text)
        _render_meta(resp)
        st.download_button(
            "Download Report (Markdown)",
            data=resp.text,
            file_name=f"{game_id}_match_report.md",
            mime="text/markdown",
            key=f"dl_report_{game_id}_{provider}_{model}",
        )
    else:
        st.caption("Generate a full tactical report — executive summary, tactics, key players, "
                   "momentum, and commentary — built only from this match's data.")


# ── Assistant tab ─────────────────────────────────────────────────────────────

def _render_assistant_tab(context, game_id: str, provider: str, model: str, configured: bool):
    if not context.has_data:
        st.info("No processed match data found for this game. Run the analysis pipeline first.")
        return

    chat_store = st.session_state.setdefault("ai_chat", {})
    history = chat_store.setdefault(game_id, [])

    header_l, header_r = st.columns([3, 1])
    with header_l:
        st.caption("Ask anything about this match — answers come only from the processed data.")
    with header_r:
        if st.button("Clear chat", width='stretch', key=f"clear_chat_{game_id}"):
            chat_store[game_id] = []
            st.rerun()

    if not history:
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        cols = st.columns(len(SUGGESTED_QUESTIONS))
        for col, q in zip(cols, SUGGESTED_QUESTIONS):
            with col:
                if st.button(q, width='stretch', disabled=not configured,
                             key=f"suggest_{game_id}_{q}"):
                    st.session_state[f"pending_q_{game_id}"] = q
                    st.rerun()

    for turn in history:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.markdown(turn["content"])
            if turn["role"] == "assistant" and turn.get("meta"):
                m = turn["meta"]
                st.markdown(
                    f"<div style='font-size:0.7rem;color:{TEXT_MUTED};'>"
                    f"{m['label']} · {m['model']} · {m['latency']:.1f}s · "
                    f"{m['tokens']} tok · {m['cost']} est.</div>",
                    unsafe_allow_html=True,
                )

    pending = st.session_state.pop(f"pending_q_{game_id}", None)
    with st.form(key=f"chat_form_{game_id}", clear_on_submit=True):
        typed = st.text_input(
            "Ask about possession, players, events, momentum...",
            key=f"chat_input_{game_id}",
            label_visibility="collapsed",
            placeholder="Ask about possession, players, events, momentum...",
        )
        submitted = st.form_submit_button("Send", type="primary", disabled=not configured)

    question = pending or (typed.strip() if submitted and typed.strip() else None)
    if not question or not configured:
        return

    client = _resolve_client(provider, model)
    if client is None:
        return

    history.append({"role": "user", "content": question})
    try:
        with st.spinner(f"Thinking with {get_spec(provider).label}..."):
            resp = answer_question(context, question, client, history[:-1])
        history.append({
            "role": "assistant",
            "content": resp.text,
            "meta": {
                "label": resp.label,
                "model": resp.model,
                "latency": resp.latency_s,
                "tokens": _fmt_tokens(resp),
                "cost": _fmt_cost(resp.cost_usd),
            },
        })
    except LLMError as exc:
        history.pop()
        st.error(str(exc))
        return

    st.rerun()


# ── Comparison tab ────────────────────────────────────────────────────────────

def _render_compare_tab(context, game_id: str):
    if not context.has_data:
        st.info("No processed match data found for this game. Run the analysis pipeline first.")
        return

    configured = [s for s in list_specs() if provider_configured(s.name)]
    st.caption("Run the same task across every configured provider and compare quality, latency, "
               "and estimated cost. Each provider uses its default model.")

    if len(configured) < 1:
        st.warning("No providers are configured. Add at least one API key to compare.")
        return
    if len(configured) < 2:
        st.info("Only one provider is configured — add a second API key (e.g. GROQ_API_KEY) "
                "to see a side-by-side comparison. You can still benchmark the one below.")

    task = st.radio(
        "Task to compare",
        options=["Match report", "Custom question"],
        horizontal=True,
        key=f"cmp_task_{game_id}",
    )
    question = ""
    if task == "Custom question":
        question = st.text_input(
            "Question",
            placeholder="e.g. Which team dominated attacks, and by how much?",
            key=f"cmp_q_{game_id}",
        ).strip()

    if not st.button("Run comparison", type="primary", key=f"cmp_run_{game_id}"):
        return
    if task == "Custom question" and not question:
        st.warning("Enter a question to compare.")
        return

    results = []  # (spec, response_or_None, error_or_None)
    progress = st.progress(0.0)
    for i, spec in enumerate(configured):
        with st.spinner(f"Running {spec.label} ({spec.default_model})..."):
            try:
                client = create_provider(spec.name, spec.default_model)
                if task == "Match report":
                    resp = generate_match_report(context, client)
                else:
                    resp = answer_question(context, question, client)
                results.append((spec, resp, None))
            except (LLMConfigError, LLMError) as exc:
                results.append((spec, None, str(exc)))
        progress.progress((i + 1) / len(configured))
    progress.empty()

    _render_comparison_results(results)


def _render_comparison_results(results):
    ok = [(spec, resp) for spec, resp, err in results if resp is not None]

    rows = []
    for spec, resp, err in results:
        if resp is not None:
            rows.append({
                "Provider": resp.label,
                "Model": resp.model,
                "Latency (s)": round(resp.latency_s, 2),
                "Tokens": resp.total_tokens if resp.total_tokens is not None else "n/a",
                "Est. cost": _fmt_cost(resp.cost_usd),
                "Words": len(resp.text.split()),
            })
        else:
            rows.append({
                "Provider": spec.label,
                "Model": spec.default_model,
                "Latency (s)": "—",
                "Tokens": "—",
                "Est. cost": "—",
                "Words": "failed",
            })

    st.markdown("<div class='section-title'>Comparison</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    if ok:
        # Highlight the fastest and cheapest, when measurable.
        fastest = min(ok, key=lambda r: r[1].latency_s)
        st.caption(f"Fastest: {fastest[1].label} at {fastest[1].latency_s:.1f}s.")
        priced = [(s, r) for s, r in ok if r.cost_usd is not None]
        if priced:
            cheapest = min(priced, key=lambda r: r[1].cost_usd)
            st.caption(f"Cheapest (estimated): {cheapest[1].label} at {_fmt_cost(cheapest[1].cost_usd)}.")

    st.markdown("<div class='section-title'>Outputs</div>", unsafe_allow_html=True)
    out_tabs = st.tabs([spec.label for spec, *_ in results])
    for tab, (spec, resp, err) in zip(out_tabs, results):
        with tab:
            if resp is not None:
                st.markdown(resp.text)
                _render_meta(resp)
            else:
                st.error(err)


# ── Entry point ───────────────────────────────────────────────────────────────

def render():
    page_header("AI Analyst", "Multi-provider match reports and a grounded Q&A assistant.")

    games = get_game_list()
    if not games:
        st.info("No games found. Run the analysis pipeline first.")
        _, r = st.columns([3, 1])
        with r:
            nav_button("Go to Analysis", "Analysis")
        return

    game_options = [(f"{g['video_name']} ({g['status']})", g['game_id']) for g in games]
    selected = st.selectbox(
        "Select Game:",
        options=game_options,
        index=0,
        format_func=lambda x: x[0],
        key="ai_game_selector",
    )
    game_id = selected[1]
    game_dir = os.path.join(INSIGHTS_DIR, game_id)

    context = _load_context(game_dir, game_id, _context_signature(game_dir))

    provider, model, configured = _provider_controls()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    tab_report, tab_chat, tab_compare = st.tabs(
        ["AI Match Report", "AI Football Assistant", "Compare Providers"]
    )
    with tab_report:
        _render_report_tab(context, game_id, provider, model, configured)
    with tab_chat:
        _render_assistant_tab(context, game_id, provider, model, configured)
    with tab_compare:
        _render_compare_tab(context, game_id)

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        nav_button("← Back to Results", "Results", key="ai_back")
    with right:
        nav_button("Home", "Home", key="ai_home")
