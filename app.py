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

# Navigation menu
with st.sidebar:
    page_name = option_menu(
        menu_title=None,
        options=[
            "Macro Data",
            "Cargo Plane Data",
            "Oil Tanker Data",
            "Stock / ETF Data",
            "Explore & Visualize",
            "Correlation Analysis",
            "Inflation-Stock Models",
            "News Sentiment",
        ],
        icons=[
            "cloud-download",
            "airplane",
            "droplet-half",
            "graph-up-arrow",
            "bar-chart-line",
            "diagram-3",
            "currency-exchange",
            "newspaper",
        ],
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

st.sidebar.markdown("---")
st.sidebar.markdown(SIDEBAR_FOOTER, unsafe_allow_html=True)

# Route to pages
if page_name == "Macro Data":
    from ui.page_download import render
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
