# dashboard/utils.py
"""Shared UI — navbar, theme, helpers."""

import streamlit as st

ACCENT       = "#dc2626"
BG_DARK      = "#030305"
BG_CARD      = "#0d0d12"
TEXT_PRIMARY  = "#f5f5f5"
TEXT_MUTED    = "#6b6b78"

NAV_PAGES = ["Home", "Upload", "Preprocess", "Analysis", "Results"]


def inject_custom_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    *, *::before, *::after {{ box-sizing: border-box; }}

    html, body, .stApp {{
        font-family: 'Inter', sans-serif !important;
        background: {BG_DARK} !important;
        color: {TEXT_PRIMARY};
    }}

    /* ── Hide ALL Streamlit chrome ───────────────────────────────── */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    header[data-testid="stHeader"],
    .stApp > header,
    header.stAppHeader,
    #MainMenu,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    .stDeployButton,
    footer {{
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        overflow: hidden !important;
    }}

    /* ── Layout ──────────────────────────────────────────────────── */
    .stApp {{ margin-top: 0 !important; padding-top: 0 !important; }}
    .stApp > .main {{ padding-top: 0 !important; margin-top: 0 !important; }}
    .block-container {{
        padding-top: 0 !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 1280px !important;
        margin: 0 auto !important;
    }}

    /* ── Navbar container ────────────────────────────────────────── */
    .ft-nav {{
        display: flex;
        align-items: center;
        height: 56px;
        padding: 0 2.5rem;
        background: rgba(6,6,10,0.98);
        border-bottom: 1px solid rgba(255,255,255,0.06);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        margin-left: -2.5rem;
        margin-right: -2.5rem;
        margin-bottom: 2.5rem;
        position: sticky;
        top: 0;
        z-index: 9999;
    }}
    .ft-nav-brand {{
        font-size: 0.9rem;
        font-weight: 800;
        color: {TEXT_PRIMARY};
        letter-spacing: -0.03em;
        flex-shrink: 0;
        margin-right: 2rem;
    }}
    .ft-nav-brand em {{
        font-style: normal;
        color: {ACCENT};
    }}
    .ft-nav-links {{
        display: flex;
        align-items: center;
        gap: 0.1rem;
        flex: 1;
        justify-content: center;
    }}
    .ft-nav-dots {{
        display: flex;
        align-items: center;
        gap: 5px;
        flex-shrink: 0;
        margin-left: 2rem;
    }}
    .ft-dot {{
        width: 6px; height: 6px;
        border-radius: 50%;
        display: inline-block;
    }}
    .ft-dot-on  {{ background: {ACCENT}; box-shadow: 0 0 6px rgba(220,38,38,0.55); }}
    .ft-dot-off {{ background: #252530; }}

    /* ── Nav buttons (inside the navbar only) ────────────────────── */
    .ft-nav .stButton > button {{
        all: unset !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0.3rem 0.88rem !important;
        border-radius: 7px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.77rem !important;
        font-weight: 500 !important;
        color: {TEXT_MUTED} !important;
        border: 1px solid transparent !important;
        cursor: pointer !important;
        white-space: nowrap !important;
        transition: color 0.15s, background 0.15s, border-color 0.15s !important;
        line-height: 1.4 !important;
    }}
    .ft-nav .stButton > button:hover {{
        color: {TEXT_PRIMARY} !important;
        background: rgba(255,255,255,0.05) !important;
    }}
    .ft-nav .stButton > button[kind="primary"] {{
        color: {TEXT_PRIMARY} !important;
        background: rgba(220,38,38,0.12) !important;
        border-color: rgba(220,38,38,0.25) !important;
    }}
    .ft-nav .stButton > button[kind="primary"]:hover {{
        background: rgba(220,38,38,0.18) !important;
    }}
    .ft-nav .stButton > button:focus {{
        outline: none !important;
        box-shadow: none !important;
    }}

    /* ── Metric card ─────────────────────────────────────────────── */
    .metric {{
        background: {BG_CARD};
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }}
    .metric-val {{
        font-size: 1.45rem; font-weight: 800; color: {ACCENT};
        line-height: 1.2; letter-spacing: -0.02em;
    }}
    .metric-label {{
        font-size: 0.62rem; color: {TEXT_MUTED};
        margin-top: 0.22rem; text-transform: uppercase; letter-spacing: 0.07em;
    }}

    /* ── Pipeline tracker ────────────────────────────────────────── */
    .pipeline {{
        display: flex; align-items: center; justify-content: center;
        margin: 0.5rem 0 2rem;
    }}
    .pipe-step {{
        background: {BG_CARD};
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 8px;
        padding: 0.45rem 1.2rem;
        text-align: center;
        min-width: 105px;
    }}
    .pipe-step.done   {{ border-color: rgba(220,38,38,0.32); background: rgba(220,38,38,0.055); }}
    .pipe-step.active {{ border-color: rgba(245,158,11,0.38); background: rgba(245,158,11,0.055); }}
    .pipe-num  {{ font-size: 0.5rem; font-weight: 700; color: {ACCENT}; text-transform: uppercase; letter-spacing: 0.1em; }}
    .pipe-step.active .pipe-num {{ color: #f59e0b; }}
    .pipe-name {{ font-size: 0.74rem; font-weight: 600; color: {TEXT_PRIMARY}; }}
    .pipe-arrow {{ color: rgba(255,255,255,0.05); font-size: 0.8rem; padding: 0 0.45rem; }}

    /* ── Card (downloads page) ───────────────────────────────────── */
    .card {{
        background: {BG_CARD};
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 14px;
        padding: 1.4rem;
        height: 100%;
        transition: border-color 0.22s, transform 0.22s, box-shadow 0.22s;
    }}
    .card:hover {{
        border-color: rgba(220,38,38,0.16);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.45);
    }}
    .card-label   {{ font-size: 0.58rem; font-weight: 700; color: {ACCENT}; text-transform: uppercase; letter-spacing: 0.13em; margin-bottom: 0.5rem; }}
    .card-heading {{ font-size: 0.95rem; font-weight: 700; color: {TEXT_PRIMARY}; margin-bottom: 0.28rem; }}
    .card-body    {{ font-size: 0.77rem; color: {TEXT_MUTED}; line-height: 1.55; }}

    /* ── Section title ───────────────────────────────────────────── */
    .section-title {{
        font-size: 0.95rem; font-weight: 700; color: {TEXT_PRIMARY};
        margin: 2rem 0 1rem; padding-bottom: 0.45rem;
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }}

    /* ── Page header ─────────────────────────────────────────────── */
    .page-header {{
        margin-bottom: 1.8rem; padding-bottom: 1.1rem;
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }}
    .page-header h1 {{
        font-size: 1.6rem; font-weight: 800; color: {TEXT_PRIMARY};
        margin: 0 0 0.18rem; letter-spacing: -0.035em;
    }}
    .page-header p {{ font-size: 0.82rem; color: {TEXT_MUTED}; margin: 0; }}

    hr {{ border-color: rgba(255,255,255,0.03) !important; }}

    /* ── Regular Streamlit buttons ───────────────────────────────── */
    .stButton > button {{
        border-radius: 8px !important; font-weight: 600 !important;
        font-size: 0.82rem !important; letter-spacing: 0.01em !important;
        padding: 0.48rem 1.4rem !important; transition: all 0.2s ease !important;
        font-family: 'Inter', sans-serif !important;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {ACCENT} 0%, #b91c1c 100%) !important;
        border: none !important; color: #fff !important;
        box-shadow: 0 2px 14px rgba(220,38,38,0.18) !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        box-shadow: 0 4px 26px rgba(220,38,38,0.38) !important;
        transform: translateY(-1px) !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        color: {TEXT_MUTED} !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        border-color: rgba(220,38,38,0.24) !important; color: {TEXT_PRIMARY} !important;
    }}

    /* ── Tabs ────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.15rem; background: transparent !important;
        border-bottom: 1px solid rgba(255,255,255,0.04) !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 7px 7px 0 0 !important; font-weight: 500 !important;
        font-size: 0.8rem !important; color: {TEXT_MUTED} !important;
        padding: 0.45rem 1rem !important; background: transparent !important; border: none !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {TEXT_PRIMARY} !important; border-bottom: 2px solid {ACCENT} !important;
    }}

    /* ── File uploader ───────────────────────────────────────────── */
    [data-testid="stFileUploader"] section {{
        background: {BG_CARD} !important;
        border: 1px dashed rgba(220,38,38,0.18) !important;
        border-radius: 12px !important;
    }}
    [data-testid="stFileUploader"] section:hover {{
        border-color: rgba(220,38,38,0.38) !important;
    }}

    /* ── Misc ────────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {{ border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.04) !important; }}
    [data-testid="stAlert"] {{ border-radius: 10px !important; border-left-width: 3px !important; }}
    [data-testid="stProgressBar"] > div > div {{ background: linear-gradient(90deg, {ACCENT}, #f87171) !important; }}
    [data-testid="stExpander"] {{ background: {BG_CARD} !important; border: 1px solid rgba(255,255,255,0.04) !important; border-radius: 10px !important; }}
    </style>
    """, unsafe_allow_html=True)


def render_navbar():
    """
    Navbar rendered as a pure HTML div (.ft-nav) with real Streamlit
    buttons inside a scoped container. The .ft-nav CSS only targets
    buttons inside itself, so it never breaks other pages.
    """
    current = st.session_state.get("page", "Home")

    has_upload     = st.session_state.get("uploaded_video") is not None
    has_preprocess = st.session_state.get("processed_video") is not None
    has_analysis   = st.session_state.get("analysis_done", False)

    dots = (
        f'<span class="ft-dot {"ft-dot-on" if has_upload else "ft-dot-off"}" title="Upload"></span>'
        f'<span class="ft-dot {"ft-dot-on" if has_preprocess else "ft-dot-off"}" title="Preprocess"></span>'
        f'<span class="ft-dot {"ft-dot-on" if has_analysis else "ft-dot-off"}" title="Analysis"></span>'
    )

    # Open the navbar div
    st.markdown(f"""
    <div class="ft-nav">
        <div class="ft-nav-brand">Football<em>Tracker</em></div>
        <div class="ft-nav-links">
    """, unsafe_allow_html=True)

    # Nav buttons — rendered inside the HTML div via Streamlit columns
    # We use a single-row columns layout that sits inside the flex container
    nav_cols = st.columns(len(NAV_PAGES))
    for col, name in zip(nav_cols, NAV_PAGES):
        with col:
            if st.button(name, key=f"nav_{name}", use_container_width=True,
                         type="primary" if current == name else "secondary"):
                st.session_state.page = name
                st.query_params["page"] = name
                st.rerun()

    # Close navbar div + dots
    st.markdown(f"""
        </div>
        <div class="ft-nav-dots">{dots}</div>
    </div>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(
        f'<div class="page-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> str:
    return (
        f'<div class="metric">'
        f'<div class="metric-val">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>'
    )


def render_pipeline(active: int = -1, done_up_to: int = -1):
    steps = ["Upload", "Preprocess", "Analysis", "Results"]
    html = '<div class="pipeline">'
    for i, name in enumerate(steps):
        if i <= done_up_to:  cls = "pipe-step done"
        elif i == active:    cls = "pipe-step active"
        else:                cls = "pipe-step"
        html += (f'<div class="{cls}">'
                 f'<div class="pipe-num">Step {i+1}</div>'
                 f'<div class="pipe-name">{name}</div></div>')
        if i < len(steps) - 1:
            html += '<div class="pipe-arrow">\u2192</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def nav_button(label: str, target: str, key: str | None = None):
    if st.button(label, use_container_width=True, type="primary", key=key):
        st.session_state.page = target
        st.query_params["page"] = target
        st.rerun()


def nav_to(page: str):
    st.session_state.page = page
    st.query_params["page"] = page
    st.rerun()


def render_site_footer():
    st.markdown(
        """
        <div style="
            margin-top: 2rem;
            padding-top: 0.9rem;
            border-top: 1px solid rgba(255,255,255,0.04);
            text-align: center;
            font-size: 0.72rem;
            color: #737380;
        ">
            Football Tracker Pipeline
        </div>
        """,
        unsafe_allow_html=True,
    )


def setup_sidebar():
    pass
