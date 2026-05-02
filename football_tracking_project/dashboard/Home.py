# dashboard/Home.py
import sys, os, base64
# Ensure project root is on sys.path so `streamlit run dashboard/Home.py` can import `dashboard` and `src` packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Football Tracker",
    page_icon=":material/sports_soccer:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from dashboard.utils import (
    inject_custom_css, render_navbar, render_site_footer, nav_button, nav_to,
    page_header, render_pipeline, setup_sidebar, metric_card,
    ACCENT, TEXT_PRIMARY, TEXT_MUTED, BG_CARD,
)

if "page" not in st.session_state:
    st.session_state.page = "Home"

# Support in-page links like ?page=Upload without changing existing button navigation.
_qp_page = st.query_params.get("page")
if isinstance(_qp_page, list):
    _qp_page = _qp_page[0] if _qp_page else None
if _qp_page in {"Home", "Upload", "Preprocess", "Analysis", "Results"}:
    st.session_state.page = _qp_page

# Heroicons-style SVGs (stroke, currentColor) for feature cards
_LP_ICONS = {
    "detection": '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M7.5 3.75H6A2.25 2.25 0 003.75 6v1.5M16.5 3.75H18A2.25 2.25 0 0120.25 6v1.5m0 9V18A2.25 2.25 0 0118 20.25h-1.5m-9 0H6A2.25 2.25 0 013.75 18v-1.5M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>',
    "tracking": '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z" /></svg>',
    "team": '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M15 19.128a9.38 9.38 0 002.625.372 9.337 9.337 0 004.121-.952 4.125 4.125 0 00-7.433-2.554M15 19.128v-.003c0-1.113-.285-2.16-.786-3.07M15 19.128v.106A12.318 12.318 0 018.624 21c-2.331 0-4.512-.645-6.374-1.766l-.001-.109a6.375 6.375 0 0111.964-3.07M12 6.375a3.375 3.375 0 11-6.75 0 3.375 3.375 0 016.75 0zm8.25 2.25a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z" /></svg>',
    "possession": '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M10.5 6a7.5 7.5 0 107.5 7.5h-7.5V6z" /><path stroke-linecap="round" stroke-linejoin="round" d="M13.5 10.5H21A7.5 7.5 0 0013.5 3v7.5z" /></svg>',
    "replay": '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="m3.375 19.5h17.25m-17.25 0A1.125 1.125 0 013.75 18.375V5.625a1.125 1.125 0 011.125-1.125h17.25A1.125 1.125 0 0121 5.625v12.75a1.125 1.125 0 01-1.125 1.125m-17.25 0V18.375m0-12.75v6.75m17.25 0V5.625m0 6.75v6.75m0 0h-5.25m5.25 0h-5.25m-5.25 0h5.25m-5.25 0v-6.75m5.25 6.75v-6.75m0 6.75H9.375m-5.25-6.75v6.75m5.25-6.75v6.75" /></svg>',
    "ball": '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M12 21a9 9 0 100-18 9 9 0 000 18z" /><path stroke-linecap="round" stroke-linejoin="round" d="M3.6 9h16.8M3.6 15h16.8M12 3a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z" /></svg>',
    "camera": '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="m15.75 10.5 4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" /></svg>',
    "export": '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" /></svg>',
}


def _team_avatar_data_uri(idx: int) -> str:
    fills = ("#1e293b", "#252e3d", "#2a3342", "#1f2836", "#243040")
    face = ("#64748b", "#6b7587", "#707a8c", "#5c6b7e", "#687288")
    f, fc = fills[idx % 5], face[idx % 5]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        f'<rect fill="{f}" width="100" height="100"/>'
        f'<circle cx="50" cy="38" r="17" fill="{fc}"/>'
        f'<ellipse cx="50" cy="95" rx="31" ry="27" fill="{fc}"/></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


_ABOUT_PLACEHOLDER_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 200" width="240" height="200">'
    '<rect fill="#0d0d12" width="240" height="200" rx="16"/>'
    '<path fill="none" stroke="#dc2626" stroke-width="2.5" opacity="0.35" d="M40 120 Q90 50 150 95"/>'
    '<circle cx="158" cy="98" r="20" fill="none" stroke="#dc2626" stroke-width="2.8"/>'
    '<path fill="none" stroke="#6b6b78" stroke-width="2" stroke-linecap="round" '
    'd="M175 75l18-12M182 68l14-20M188 82l22-6"/>'
    "</svg>"
)


def _about_us_component_html(image_data_url: str) -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
  <style>
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      padding: 0;
      background: #030305;
      font-family: 'Inter', system-ui, sans-serif;
      color: #f5f5f5;
      -webkit-font-smoothing: antialiased;
    }
    .about-root {
      max-width: 1120px;
      margin: 0 auto;
      padding: 0.5rem 1.25rem 0.25rem;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1.05fr);
      gap: clamp(1.25rem, 4vw, 2.75rem);
      align-items: center;
    }
    @media (max-width: 820px) {
      .about-root { grid-template-columns: 1fr; }
    }
    .about-visual {
      position: relative;
    }
    .about-frame {
      position: relative;
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.08);
      background: #101018;
      aspect-ratio: 4 / 3;
      box-shadow: 0 20px 50px rgba(0,0,0,0.45);
    }
    .about-frame:not(.is-loaded) {
      animation: aboutFramePulse 1.15s ease-in-out infinite;
    }
    @keyframes aboutFramePulse {
      0%, 100% { box-shadow: 0 16px 40px rgba(0,0,0,0.4), 0 0 0 0 rgba(220,38,38,0); }
      50% { box-shadow: 0 20px 52px rgba(0,0,0,0.5), 0 0 28px 1px rgba(220,38,38,0.12); }
    }
    .about-shimmer {
      position: absolute;
      inset: 0;
      z-index: 2;
      background: linear-gradient(
        105deg,
        transparent 0%,
        rgba(255,255,255,0.04) 45%,
        rgba(255,255,255,0.09) 50%,
        rgba(255,255,255,0.04) 55%,
        transparent 100%
      );
      background-size: 220% 100%;
      animation: aboutShimmer 1.1s linear infinite;
      pointer-events: none;
      border-radius: inherit;
    }
    @keyframes aboutShimmer {
      0% { background-position: 120% 0; }
      100% { background-position: -120% 0; }
    }
    .about-frame.is-loaded .about-shimmer {
      opacity: 0;
      visibility: hidden;
      transition: opacity 0.5s ease, visibility 0.5s;
    }
    .about-photo {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: center 35%;
      opacity: 0;
      transform: scale(1.03);
      transition: opacity 0.55s ease, transform 0.65s ease;
    }
    .about-frame.is-loaded .about-photo {
      opacity: 1;
      transform: scale(1);
      animation: aboutFloat 5s ease-in-out infinite;
      animation-delay: 0.2s;
    }
    @keyframes aboutFloat {
      0%, 100% { transform: translateY(0) scale(1); }
      50% { transform: translateY(-7px) scale(1.01); }
    }
    .about-eyebrow {
      font-size: 0.55rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: #6b6b78;
      margin: 0 0 0.5rem;
    }
    .about-h2 {
      font-size: clamp(1.15rem, 2.4vw, 1.45rem);
      font-weight: 800;
      letter-spacing: -0.03em;
      margin: 0 0 0.85rem;
      line-height: 1.2;
    }
    .about-h2 em {
      font-style: normal;
      background: linear-gradient(135deg, #dc2626, #f87171);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .about-p {
      font-size: 0.82rem;
      line-height: 1.65;
      color: #9a9aa8;
      margin: 0 0 0.65rem;
    }
    .about-p:last-child { margin-bottom: 0; }
  </style>
</head>
<body>
  <div class="about-root">
    <div class="about-visual">
      <div class="about-frame" id="aboutImgFrame">
        <div class="about-shimmer" aria-hidden="true"></div>
        <img class="about-photo" src="__SRC__" alt="Football match action"
             onload="document.getElementById('aboutImgFrame').classList.add('is-loaded')"
             onerror="document.getElementById('aboutImgFrame').classList.add('is-loaded')"/>
      </div>
    </div>
    <div class="about-copy">
      <p class="about-eyebrow">About us</p>
      <h2 class="about-h2">Built for the <em>beautiful game</em></h2>
      <p class="about-p">
        Football Tracker turns raw match video into structured data: players, ball, kits,
        possession, and exports you can plug into analysis tools or broadcast workflows.
      </p>
      <p class="about-p">
        We combine modern detection and tracking with a simple Streamlit pipeline so coaches,
        analysts, and researchers can go from upload to insight without a heavy stack.
      </p>
    </div>
  </div>
</body>
</html>
""".replace("__SRC__", image_data_url)


def _render_about_us(image_data_url: str) -> None:
    components.html(
        _about_us_component_html(image_data_url),
        height=460,
        scrolling=False,
    )


inject_custom_css()
setup_sidebar()
render_navbar()

current = st.session_state.get("page", "Home")

if current == "Upload":
    from dashboard.pages import upload_page; upload_page.render()
elif current == "Preprocess":
    from dashboard.pages import preprocess_page; preprocess_page.render()
elif current == "Analysis":
    from dashboard.pages import analysis_page; analysis_page.render()
elif current == "Results":
    from dashboard.pages import results_page; results_page.render()
else:
    # ══════════════════════════════════════════════════════════════════
    #  LANDING PAGE
    # ══════════════════════════════════════════════════════════════════

    # Load hero image
    _IMG_B64 = ""
    _img_path = os.path.join(os.path.dirname(__file__), "images", "images.jpeg")
    if os.path.exists(_img_path):
        with open(_img_path, "rb") as f:
            _IMG_B64 = base64.b64encode(f.read()).decode()

    # Load about image (bottom section image)
    _ABOUT_IMG_B64 = ""
    _about_img_path = os.path.join(os.path.dirname(__file__), "images", "image.png")
    if os.path.exists(_about_img_path):
        with open(_about_img_path, "rb") as f:
            _ABOUT_IMG_B64 = base64.b64encode(f.read()).decode()

    # All landing-page classes use "lp-" prefix to avoid any conflict
    # with the shared CSS in utils.py
    st.markdown("""
    <style>
    .lp-hero {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        min-height: 420px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-size: cover;
        background-position: center 30%;
        margin-bottom: 2rem;
    }
    .lp-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 16px;
        background: linear-gradient(155deg,
            rgba(3,3,5,0.94) 0%,
            rgba(8,6,8,0.85) 50%,
            rgba(12,4,6,0.92) 100%);
    }
    .lp-hero::after {
        content: "";
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 2px;
        border-radius: 0 0 16px 16px;
        background: linear-gradient(90deg, transparent 10%, rgba(220,38,38,0.7) 50%, transparent 90%);
        opacity: 0.85;
    }
    .lp-inner {
        position: relative;
        z-index: 2;
        text-align: center;
        padding: 3.5rem clamp(1.25rem, 4vw, 2.5rem) 3.25rem;
        width: 100%;
        max-width: 720px;
        margin: 0 auto;
    }
    .lp-badge {
        display: inline-block;
        font-size: 0.54rem;
        font-weight: 700;
        color: rgba(245,245,245,0.55);
        text-transform: uppercase;
        letter-spacing: 0.18em;
        padding: 0.24rem 0.85rem;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 999px;
        margin-bottom: 1rem;
        background: rgba(255,255,255,0.04);
    }
    .lp-h1 {
        font-size: clamp(2.25rem, 5vw, 3.25rem);
        font-weight: 800;
        color: #f5f5f5;
        letter-spacing: -0.045em;
        line-height: 1.08;
        margin: 0 0 1.1rem;
        text-shadow: 0 2px 24px rgba(0,0,0,0.45);
    }
    .lp-h1 em {
        font-style: normal;
        background: linear-gradient(135deg, #dc2626 0%, #f87171 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .lp-hero-icons {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.55rem 0.65rem;
        margin: 0.85rem auto 0;
        max-width: 420px;
    }
    .lp-hero-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        color: #dc2626;
    }
    .lp-hero-icon svg { width: 20px; height: 20px; }
    .lp-hero-enter {
        display: inline-block;
        position: absolute;
        right: clamp(1rem, 3vw, 2rem);
        bottom: 1rem;
        margin: 0;
        padding: 0.48rem 1.05rem;
        border-radius: 9px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        color: #f5f5f5 !important;
        text-decoration: none !important;
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        border: 1px solid rgba(220,38,38,0.35);
        box-shadow: 0 2px 16px rgba(220,38,38,0.22);
        z-index: 2;
    }

    /* Hero + CTA row (headline | Enter pipeline) */
    .main .block-container [data-testid="stHorizontalBlock"]:has(.lp-hero--split) {
        gap: 0 !important;
        align-items: stretch !important;
    }
    .main .block-container [data-testid="stHorizontalBlock"]:has(.lp-hero--split) > div[data-testid="column"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    .lp-hero--split {
        margin-bottom: 0 !important;
        min-height: 400px;
        border-radius: 16px 0 0 16px;
        display: flex;
        align-items: center;
        justify-content: flex-start;
    }
    .lp-inner--split {
        position: relative;
        text-align: left !important;
        margin: 0 !important;
        max-width: none !important;
        width: 100%;
        padding: 2.5rem clamp(1rem, 3vw, 2rem) 4.35rem clamp(1.25rem, 3.5vw, 2.25rem) !important;
    }
    .lp-inner--split .lp-h1 {
        margin-bottom: 0.85rem !important;
    }
    .lp-inner--split .lp-hero-icons {
        justify-content: flex-start;
        margin-left: 0;
        margin-top: 0.75rem;
        max-width: none;
    }
    .main .block-container [data-testid="stHorizontalBlock"]:has(.lp-hero--split) > div[data-testid="column"]:nth-child(2) {
        display: flex !important;
        flex-direction: column !important;
        align-items: stretch !important;
        justify-content: center !important;
        min-height: 400px;
        background: linear-gradient(180deg, rgba(7,7,12,0.96), rgba(4,4,8,0.99));
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 1px solid rgba(255,255,255,0.05);
        border-radius: 0 16px 16px 0;
        position: relative;
        padding: 1rem 0.75rem !important;
        box-sizing: border-box;
    }
    .main .block-container [data-testid="stHorizontalBlock"]:has(.lp-hero--split) > div[data-testid="column"]:nth-child(2)::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent 15%, rgba(220,38,38,0.55) 50%, transparent 85%);
        opacity: 0.85;
        border-radius: 0 0 16px 0;
    }
    .lp-hero--split::after {
        border-radius: 0 0 0 16px;
    }
    @media (max-width: 640px) {
        .lp-hero--split {
            min-height: 340px;
            border-radius: 16px 16px 0 0;
        }
        .main .block-container [data-testid="stHorizontalBlock"]:has(.lp-hero--split) > div[data-testid="column"]:nth-child(2) {
            min-height: auto;
            border-radius: 0 0 16px 16px;
            border-left: 1px solid rgba(255,255,255,0.08);
        }
        .main .block-container [data-testid="stHorizontalBlock"]:has(.lp-hero--split) > div[data-testid="column"]:nth-child(2)::after {
            border-radius: 0 0 16px 16px;
        }
    }

    /* Section headings */
    .lp-label {
        text-align: center;
        font-size: 0.52rem;
        font-weight: 700;
        color: #6b6b78;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        margin-bottom: 0.45rem;
    }
    .lp-h2 {
        text-align: center;
        font-size: clamp(1.05rem, 2.2vw, 1.28rem);
        font-weight: 700;
        color: #f5f5f5;
        letter-spacing: -0.03em;
        margin-bottom: 1.65rem;
        line-height: 1.3;
    }
    .lp-h2 em {
        font-style: normal;
        background: linear-gradient(135deg, #dc2626, #f87171);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Pipeline: full-width sponsor-style infinite marquee */
    .lp-marquee-outer {
        width: 100vw;
        margin-left: calc(50% - 50vw);
        margin-right: calc(50% - 50vw);
        margin-bottom: 0.5rem;
        position: relative;
    }
    .lp-marquee-viewport {
        overflow: hidden;
        width: 100%;
        padding: 0.6rem 0 1rem;
        mask-image: linear-gradient(90deg, transparent 0%, #000 6%, #000 94%, transparent 100%);
        -webkit-mask-image: linear-gradient(90deg, transparent 0%, #000 6%, #000 94%, transparent 100%);
    }
    .lp-marquee-track {
        display: flex;
        flex-direction: row;
        flex-wrap: nowrap;
        width: max-content;
        gap: 14px;
        animation: lpMarqueeScroll 55s linear infinite;
    }
    .lp-marquee-item {
        flex: 0 0 auto;
        box-sizing: border-box;
        /* ~6+ cards visible on a typical laptop viewport */
        width: clamp(118px, calc((100vw - 56px) / 6.15), 200px);
    }
    @media (max-width: 900px) {
        .lp-marquee-item {
            width: clamp(118px, calc((100vw - 40px) / 4.1), 200px);
        }
    }
    @media (max-width: 520px) {
        .lp-marquee-item {
            width: clamp(110px, 42vw, 160px);
        }
    }
    .lp-marquee-card {
        height: 100%;
        min-height: 132px;
        padding: 1rem 0.65rem 1.1rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(165deg, #12121a 0%, #0a0a0f 100%);
        text-align: center;
        box-shadow: 0 8px 28px rgba(0,0,0,0.3);
    }
    .lp-mq-icon {
        width: 48px;
        height: 48px;
        margin: 0 auto 0.55rem;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(220,38,38,0.08);
        border: 1px solid rgba(220,38,38,0.16);
        color: #dc2626;
    }
    .lp-mq-icon svg { width: 24px; height: 24px; }
    .lp-mq-title {
        font-size: 0.72rem;
        font-weight: 600;
        color: #f5f5f5;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.25;
    }
    @keyframes lpMarqueeScroll {
        from { transform: translateX(0); }
        to { transform: translateX(-50%); }
    }
    @media (prefers-reduced-motion: reduce) {
        .lp-marquee-track {
            animation: none !important;
            transform: translateX(0) !important;
        }
    }

    /* Meet the team */
    /* About us (outer spacing; iframe body styles are inside component) */
    .lp-about-spacer {
        height: 2.25rem;
    }

    .lp-team-wrap {
        margin: 2.75rem 0 2rem;
        padding: 2rem clamp(0.5rem, 2vw, 1rem);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.06);
        background: #0a0a0f;
    }
    .lp-team-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1.1rem;
        max-width: 920px;
        margin: 0 auto;
        align-items: start;
    }
    @media (max-width: 900px) {
        .lp-team-grid { grid-template-columns: repeat(3, 1fr); }
    }
    @media (max-width: 520px) {
        .lp-team-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .lp-team-card {
        text-align: center;
    }
    .lp-team-photo {
        width: 112px;
        height: 112px;
        border-radius: 50%;
        object-fit: cover;
        object-position: center;
        margin: 0 auto 0.65rem;
        display: block;
        border: 2px solid rgba(220,38,38,0.2);
        box-shadow: 0 6px 20px rgba(0,0,0,0.35);
    }
    .lp-team-name {
        font-size: 0.8rem;
        font-weight: 700;
        color: #f5f5f5;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .lp-team-role {
        font-size: 0.65rem;
        color: #6b6b78;
        line-height: 1.35;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    </style>
    """, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────
    if _IMG_B64:
        bg = f"background-image:url('data:image/jpeg;base64,{_IMG_B64}');"
    else:
        bg = "background:linear-gradient(160deg,#07070c,#0d0d12 50%,#0a0008);"

    _icon_order = (
        "detection",
        "tracking",
        "team",
        "possession",
        "replay",
        "ball",
        "camera",
        "export",
    )
    _hero_icons_html = "".join(
        f'<span class="lp-hero-icon" aria-hidden="true">{_LP_ICONS[k]}</span>'
        for k in _icon_order
    )

    st.markdown(
        f"""
        <div class="lp-hero lp-hero--split" style="{bg}">
            <div class="lp-inner lp-inner--split">
                <a class="lp-hero-enter" href="?page=Upload" target="_self" rel="noopener">Enter pipeline</a>
                <div class="lp-badge">Match video</div>
                <h1 class="lp-h1">From kickoff<br><em>to data</em></h1>
                <div class="lp-hero-icons">{_hero_icons_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)

    st.markdown('<div class="lp-about-spacer"></div>', unsafe_allow_html=True)
    if _ABOUT_IMG_B64:
        _about_url = f"data:image/png;base64,{_ABOUT_IMG_B64}"
    else:
        _about_url = (
            "data:image/svg+xml;base64,"
            + base64.b64encode(_ABOUT_PLACEHOLDER_SVG.encode()).decode()
        )
    _render_about_us(_about_url)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="lp-h2" style="margin-bottom:1rem">Pipeline</div>',
        unsafe_allow_html=True,
    )

    _slides = [
        ("detection", "Detection"),
        ("tracking", "Tracking"),
        ("team", "Team split"),
        ("possession", "Possession"),
        ("replay", "Replays"),
        ("ball", "Ball gaps"),
        ("camera", "Camera"),
        ("export", "Export"),
    ]

    def _mq_item(_ik: str, _title: str) -> str:
        _svg = _LP_ICONS[_ik]
        return (
            '<div class="lp-marquee-item">'
            '<div class="lp-marquee-card">'
            f'<div class="lp-mq-icon" aria-hidden="true">{_svg}</div>'
            f'<p class="lp-mq-title">{_title}</p>'
            "</div></div>"
        )

    _mq_row = "".join(_mq_item(ik, t) for ik, t in _slides)
    _car_html = (
        '<div class="lp-marquee-outer" aria-label="Pipeline capabilities">'
        '<div class="lp-marquee-viewport">'
        '<div class="lp-marquee-track">'
        + _mq_row
        + _mq_row
        + "</div></div></div>"
    )
    st.markdown(_car_html, unsafe_allow_html=True)

    _team = [
        ("tse-coder", "Core contributor"),
        ("BytePhilosopher", "Core contributor"),
        ("yoseph404", "Core contributor"),
        ("halafiCodes", "Core contributor"),
        ("abeladamushumet", "Core contributor"),
    ]
    _cards = "".join(
        f'<div class="lp-team-card">'
        f'<img class="lp-team-photo" src="https://github.com/{name}.png" width="112" height="112" alt="{name}" />'
        f'<div class="lp-team-name">{name}</div>'
        f'<div class="lp-team-role">{role}</div>'
        f"</div>"
        for name, role in _team
    )
    st.markdown(
        f"""
        <div class="lp-team-wrap">
          <div class="lp-label">Squad</div>
          <div class="lp-h2">Meet the <em>team</em></div>
          <div class="lp-team-grid">{_cards}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_site_footer()
