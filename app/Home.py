# app/Home.py
import os
import sys

# Ensure project root is on sys.path so `streamlit run app/Home.py` can import `app` and `src` packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.pages import home_page
from app.utils import inject_custom_css, render_navbar, render_site_footer, setup_sidebar


st.set_page_config(
    page_title="Football Tracker",
    page_icon=":material/sports_soccer:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


if "page" not in st.session_state:
    st.session_state.page = "Home"

# Support in-page links like ?page=Upload without changing existing button navigation.
_qp_page = st.query_params.get("page")
if isinstance(_qp_page, list):
    _qp_page = _qp_page[0] if _qp_page else None
if _qp_page in {"Home", "Upload", "Preprocess", "Analysis", "Results"}:
    st.session_state.page = _qp_page


inject_custom_css()
setup_sidebar()
render_navbar()

current = st.session_state.get("page", "Home")

if current == "Upload":
    from app.pages import upload_page

    upload_page.render()
elif current == "Preprocess":
    from app.pages import preprocess_page

    preprocess_page.render()
elif current == "Analysis":
    from app.pages import analysis_page

    analysis_page.render()
elif current == "Results":
    from app.pages import results_page

    results_page.render()
else:
    home_page.render()

render_site_footer()
