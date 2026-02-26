"""Economic Simulation & Correlation Analysis Dashboard.

Run with: streamlit run app.py
"""

import logging
logging.getLogger("torch.classes").setLevel(logging.ERROR)

import streamlit as st
from streamlit_option_menu import option_menu
from data_sources.database import init_db

init_db()

st.set_page_config(
    page_title="Econ Sim - Data Intelligence",
    page_icon="\u26A1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject futuristic blue theme CSS and register Plotly template
from ui.theme import STEAM_CSS, HEADER_BANNER, SIDEBAR_HEADER, SIDEBAR_FOOTER  # noqa: E402

st.markdown(STEAM_CSS, unsafe_allow_html=True)
st.markdown(HEADER_BANNER, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown(SIDEBAR_HEADER, unsafe_allow_html=True)
st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Determine whether data already exists so we can hide download pages
# ---------------------------------------------------------------------------
from data_sources.world_bank import list_saved_datasets  # noqa: E402

_has_data = bool(list_saved_datasets()) or "current_dataset" in st.session_state

# Data-download pages and their icons
_DATA_PAGES = [
    ("Macro Data", "cloud-download"),
    ("USA Economy (FRED)", "flag"),
    ("Cargo Plane Data", "airplane"),
    ("Oil Tanker Data", "droplet-half"),
    ("Stock / ETF Data", "graph-up-arrow"),
]

# Analysis / visualisation pages (always visible)
_ANALYSIS_PAGES = [
    ("Explore & Visualize", "bar-chart-line"),
    ("Correlation Analysis", "diagram-3"),
    ("Inflation-Stock Models", "currency-exchange"),
    ("News Sentiment", "newspaper"),
]

# Toggle for showing data download pages when data already exists
with st.sidebar:
    if _has_data:
        show_data_pages = st.toggle(
            "Show data download pages",
            value=False,
            key="show_data_pages",
            help="Data has already been downloaded. Toggle on to access the download pages again.",
        )
    else:
        show_data_pages = True  # no data yet — always show

# Build the dynamic menu
if show_data_pages:
    _visible_pages = _DATA_PAGES + _ANALYSIS_PAGES
else:
    _visible_pages = _ANALYSIS_PAGES

_page_options = [p[0] for p in _visible_pages]
_page_icons = [p[1] for p in _visible_pages]

# Navigation menu
with st.sidebar:
    page_name = option_menu(
        menu_title=None,
        options=_page_options,
        icons=_page_icons,
        default_index=0,
        styles={
            "container": {
                "padding": "0",
                "background-color": "transparent",
            },
            "icon": {
                "color": "#38BDF8",
                "font-size": "16px",
            },
            "nav-link": {
                "font-family": "'Inter', sans-serif",
                "font-size": "14px",
                "font-weight": "400",
                "color": "#94A3B8",
                "text-align": "left",
                "padding": "10px 16px",
                "margin": "2px 0",
                "border-radius": "8px",
                "background-color": "transparent",
                "border": "1px solid transparent",
                "transition": "all 0.3s ease",
                "--hover-color": "rgba(0,180,216,0.08)",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, rgba(0,180,216,0.15) 0%, rgba(129,140,248,0.12) 100%)",
                "color": "#00B4D8",
                "font-weight": "500",
                "border": "1px solid rgba(0,180,216,0.3)",
                "box-shadow": "0 0 15px rgba(0,180,216,0.1)",
            },
        },
    )

# ---------------------------------------------------------------------------
# Quick-download buttons (shown when data pages are hidden)
# ---------------------------------------------------------------------------
if _has_data and not show_data_pages:
    with st.sidebar:
        st.markdown("---")
        with st.expander("Quick Download Data", expanded=False):
            st.caption("Download new data without leaving the current page.")

            if st.button("Macro Data", key="qd_macro", use_container_width=True):
                st.session_state["_qd_target"] = "Macro Data"
                st.session_state["show_data_pages"] = True
                st.rerun()

            if st.button("USA Economy (FRED)", key="qd_fred", use_container_width=True):
                st.session_state["_qd_target"] = "USA Economy (FRED)"
                st.session_state["show_data_pages"] = True
                st.rerun()

            if st.button("Cargo Plane Data", key="qd_cargo", use_container_width=True):
                st.session_state["_qd_target"] = "Cargo Plane Data"
                st.session_state["show_data_pages"] = True
                st.rerun()

            if st.button("Oil Tanker Data", key="qd_oil", use_container_width=True):
                st.session_state["_qd_target"] = "Oil Tanker Data"
                st.session_state["show_data_pages"] = True
                st.rerun()

            if st.button("Stock / ETF Data", key="qd_stock", use_container_width=True):
                st.session_state["_qd_target"] = "Stock / ETF Data"
                st.session_state["show_data_pages"] = True
                st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(SIDEBAR_FOOTER, unsafe_allow_html=True)

# Route to pages
if page_name == "Macro Data":
    from ui.page_download import render
    render()
elif page_name == "USA Economy (FRED)":
    from ui.page_usa_economy import render
    render()
elif page_name == "Stock / ETF Data":
    from ui.page_stock_download import render
    render()
elif page_name == "Explore & Visualize":
    from ui.page_explore import render
    render()
elif page_name == "Correlation Analysis":
    from ui.page_correlations import render
    render()
elif page_name == "Inflation-Stock Models":
    from ui.page_inflation_stock import render
    render()
elif page_name == "Cargo Plane Data":
    from ui.page_cargo import render
    render()
elif page_name == "Oil Tanker Data":
    from ui.page_oil_tankers import render
    render()
elif page_name == "News Sentiment":
    from ui.page_news_sentiment import render
    render()
