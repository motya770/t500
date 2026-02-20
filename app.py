"""Economic Simulation & Correlation Analysis Dashboard.

Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Econ Express - Steam Train Dashboard",
    page_icon="\U0001F682",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject steam train theme CSS and register Plotly template
from ui.theme import STEAM_CSS, HEADER_BANNER, SIDEBAR_HEADER, SIDEBAR_FOOTER  # noqa: E402

st.markdown(STEAM_CSS, unsafe_allow_html=True)
st.markdown(HEADER_BANNER, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown(SIDEBAR_HEADER, unsafe_allow_html=True)
st.sidebar.markdown("---")

# Train-themed navigation labels
page = st.sidebar.radio(
    "\U0001F6E4\uFE0F Route Select",
    [
        "\U0001F4E6 Download Data",
        "\U0001F4C8 Stock / ETF Data",
        "\U0001F50D Explore & Visualize",
        "\U0001F517 Correlation Analysis",
        "\U0001F4CA Inflation-Stock Models",
        "\u2708 Cargo Plane Analysis",
        "\U0001F4F0 News Sentiment",
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
elif page_name == "News Sentiment":
    from ui.page_news_sentiment import render
    render()
