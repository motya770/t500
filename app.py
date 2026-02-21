"""Economic Simulation & Correlation Analysis Dashboard.

Run with: streamlit run app.py
"""

import streamlit as st
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

# Navigation
page = st.sidebar.radio(
    "\u25C8 Navigate",
    [
        "\u25B8 Download Data",
        "\u25B8 Stock / ETF Data",
        "\u25B8 Explore & Visualize",
        "\u25B8 Correlation Analysis",
        "\u25B8 Inflation-Stock Models",
        "\u25B8 Cargo Plane Analysis",
        "\u25B8 Oil Tanker Analysis",
        "\u25B8 News Sentiment",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(SIDEBAR_FOOTER, unsafe_allow_html=True)

# Route to pages (strip emoji prefix for matching)
page_name = page.split(" ", 1)[1] if " " in page else page

if page_name == "Download Data":
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
elif page_name == "Cargo Plane Analysis":
    from ui.page_cargo import render
    render()
elif page_name == "Oil Tanker Analysis":
    from ui.page_oil_tankers import render
    render()
elif page_name == "News Sentiment":
    from ui.page_news_sentiment import render
    render()
