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
    ("Global Financial Markets", "currency-exchange"),
    ("Global Monetary & Bonds", "bank"),
    ("Global Sentiment & Trade", "graph-up"),
]

# Analysis / visualisation pages (always visible)
_ANALYSIS_PAGES = [
    ("Stock Analyses", "currency-exchange"),
    ("Correlation Analysis", "diagram-3"),
    ("Explore & Visualize", "bar-chart-line"),
    ("News Sentiment", "newspaper"),
    ("Investigate Data", "table"),
]

# Read toggle value from session state BEFORE building the menu
# (the actual toggle widget is rendered AFTER the menu)
if _has_data:
    show_data_pages = st.session_state.get("show_data_pages", False)
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

# Toggle for data-download pages (below the menu)
with st.sidebar:
    if _has_data:
        st.markdown("---")
        show_data_pages = st.toggle(
            "Show data download pages",
            value=False,
            key="show_data_pages",
            help="Data has already been downloaded. Toggle on to access the download pages again.",
        )

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
elif page_name == "Investigate Data":
    from ui.page_investigate import render
    render()
elif page_name == "Explore & Visualize":
    from ui.page_explore import render
    render()
elif page_name == "Correlation Analysis":
    from ui.page_correlations import render
    render()
elif page_name == "Stock Analyses":
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
elif page_name == "Global Financial Markets":
    from ui.page_global_markets import render
    render()
elif page_name == "Global Monetary & Bonds":
    from ui.page_global_monetary import render
    render()
elif page_name == "Global Sentiment & Trade":
    from ui.page_global_sentiment import render
    render()
