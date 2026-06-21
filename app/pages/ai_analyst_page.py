# app/pages/ai_analyst_page.py
"""AI Analyst Page — Gemini-powered match report and grounded chat assistant.

Consumes the existing pipeline outputs for a selected game and exposes two tabs:
an on-demand natural-language match report and a question-answering assistant.
Both are grounded exclusively in the processed match data.
"""

import os

import streamlit as st

from app.config import INSIGHTS_DIR, get_game_list
from app.utils import page_header, nav_button, TEXT_MUTED
from src.exporters.output_schema import OutputFiles

from ai.data_loader import build_match_context
from ai.gemini_client import GeminiClient, GeminiConfigError, GeminiError
from ai.match_report import generate_match_report
from ai.chat_assistant import answer_question

EVENTS_JSON = "events.json"

SUGGESTED_QUESTIONS = [
    "Which team controlled possession?",
    "Who covered the most distance?",
    "What was the fastest sprint?",
    "Which team dominated attacks?",
    "Summarize the match.",
]


@st.cache_resource(show_spinner=False)
def _get_client() -> GeminiClient:
    return GeminiClient()


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


def _render_api_key_help():
    st.warning("Gemini is not configured.")
    st.markdown(
        f"""
        <div style="font-size:0.85rem; color:{TEXT_MUTED}; line-height:1.6;">
        Set a <code>GEMINI_API_KEY</code> environment variable (or add it to a
        <code>.env</code> file in the project root), then reload this page:
        <pre style="margin-top:0.5rem;">GEMINI_API_KEY=your_key_here</pre>
        Get a key from Google AI Studio.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_report_tab(context, game_id: str):
    if not context.has_data:
        st.info("No processed match data found for this game. Run the analysis pipeline first.")
        return

    report_key = f"ai_report_{game_id}"
    existing = st.session_state.get(report_key)

    c1, c2 = st.columns([1, 1])
    with c1:
        generate = st.button("Generate AI Match Report", type="primary",
                             width='stretch', key=f"gen_report_{game_id}")
    with c2:
        if existing and st.button("Regenerate", width='stretch', key=f"regen_report_{game_id}"):
            st.session_state.pop(report_key, None)
            generate = True

    if generate:
        try:
            client = _get_client()
        except GeminiConfigError:
            _render_api_key_help()
            return
        try:
            with st.spinner("Analyzing match data with Gemini..."):
                st.session_state[report_key] = generate_match_report(context, client)
        except GeminiError as exc:
            st.error(str(exc))
            return

    report = st.session_state.get(report_key)
    if report:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown(report)
        st.download_button(
            "Download Report (Markdown)",
            data=report,
            file_name=f"{game_id}_match_report.md",
            mime="text/markdown",
            key=f"dl_report_{game_id}",
        )
    else:
        st.caption("Generate a full tactical report — executive summary, tactics, key players, "
                   "momentum, and commentary — built only from this match's data.")


def _render_assistant_tab(context, game_id: str):
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
                if st.button(q, width='stretch', key=f"suggest_{game_id}_{q}"):
                    st.session_state[f"pending_q_{game_id}"] = q
                    st.rerun()

    for turn in history:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.markdown(turn["content"])

    pending = st.session_state.pop(f"pending_q_{game_id}", None)
    with st.form(key=f"chat_form_{game_id}", clear_on_submit=True):
        typed = st.text_input(
            "Ask about possession, players, events, momentum...",
            key=f"chat_input_{game_id}",
            label_visibility="collapsed",
            placeholder="Ask about possession, players, events, momentum...",
        )
        submitted = st.form_submit_button("Send", type="primary")

    question = pending or (typed.strip() if submitted and typed.strip() else None)
    if not question:
        return

    try:
        client = _get_client()
    except GeminiConfigError:
        _render_api_key_help()
        return

    history.append({"role": "user", "content": question})
    try:
        with st.spinner("Thinking..."):
            answer = answer_question(context, question, client, history[:-1])
        history.append({"role": "assistant", "content": answer})
    except GeminiError as exc:
        history.pop()
        st.error(str(exc))
        return

    st.rerun()


def render():
    page_header("AI Analyst", "Gemini-powered match reports and a grounded Q&A assistant.")

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

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    tab_report, tab_chat = st.tabs(["AI Match Report", "AI Football Assistant"])
    with tab_report:
        _render_report_tab(context, game_id)
    with tab_chat:
        _render_assistant_tab(context, game_id)

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        nav_button("← Back to Results", "Results", key="ai_back")
    with right:
        nav_button("Home", "Home", key="ai_home")
