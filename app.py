"""Economic Simulation & Correlation Analysis Dashboard.

Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Economic Simulation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("Economic Simulation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Download Data", "Explore & Visualize", "Correlation Analysis", "Cargo Plane Analysis"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data source:** [World Bank Open Data](https://data.worldbank.org/)\n\n"
    "**Methods:** Pearson, Spearman, Kendall, Partial Correlation, "
    "Mutual Information, Random Forest, Gradient Boosting, Lasso, "
    "Elastic Net, PCA, Autoencoder, Granger Causality\n\n"
    "**Cargo Analysis:** Freight trends, rankings, YoY growth, "
    "cargo intensity, economic correlations, ML growth drivers"
)

# Route to pages
if page == "Download Data":
    from ui.page_download import render
    render()
elif page == "Explore & Visualize":
    from ui.page_explore import render
    render()
elif page == "Correlation Analysis":
    from ui.page_correlations import render
    render()
elif page == "Cargo Plane Analysis":
    from ui.page_cargo import render
    render()
