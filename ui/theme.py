"""Steam Train visual theme for the Economic Simulation dashboard.

Provides a consistent color palette, CSS injection, and Plotly chart styling
inspired by the industrial steam age: brass, copper, iron, mahogany, and coal.
"""

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COAL = "#1B1B1B"
DARK_IRON = "#2A2A2A"
IRON = "#4A4A4A"
STEEL = "#8B8B8B"
MAHOGANY = "#3C1414"
DARK_WOOD = "#2A2016"
WOOD = "#5C3A1E"
BRASS = "#B87333"
COPPER = "#CD7F32"
BRONZE = "#D4A574"
GOLD = "#DAA520"
EMBER = "#E25822"
FIRE_ORANGE = "#FF8C00"
CREAM = "#E8DCC8"
PARCHMENT = "#FFF8E7"
STEAM = "#C0C0C0"
SMOKE = "#A9A9A9"

# Plotly chart color sequence (warm industrial tones)
CHART_COLORS = [
    "#B87333",  # copper
    "#E25822",  # ember red
    "#DAA520",  # goldenrod
    "#CD853F",  # peru/bronze
    "#8B4513",  # saddle brown
    "#D2691E",  # chocolate
    "#A0522D",  # sienna
    "#708090",  # slate gray (steel)
    "#BC8F8F",  # rosy brown
    "#F4A460",  # sandy brown
    "#C0C0C0",  # silver
    "#FF8C00",  # dark orange
]

# Heatmap color scales
HEATMAP_SCALE = [
    [0.0, "#1B1B1B"],
    [0.15, "#3C1414"],
    [0.3, "#8B4513"],
    [0.5, "#B87333"],
    [0.7, "#DAA520"],
    [0.85, "#F4A460"],
    [1.0, "#FFF8E7"],
]

DIVERGING_SCALE = [
    [0.0, "#1B3A5C"],
    [0.25, "#4A708B"],
    [0.5, "#E8DCC8"],
    [0.75, "#CD7F32"],
    [1.0, "#8B2500"],
]

# ---------------------------------------------------------------------------
# Plotly layout template
# ---------------------------------------------------------------------------
_STEAM_LAYOUT = go.Layout(
    paper_bgcolor=COAL,
    plot_bgcolor=DARK_IRON,
    font=dict(family="Georgia, serif", color=CREAM, size=13),
    title=dict(font=dict(family="Georgia, serif", color=BRASS, size=18)),
    xaxis=dict(
        gridcolor=IRON,
        linecolor=STEEL,
        tickfont=dict(color=STEAM),
        title_font=dict(color=BRONZE),
    ),
    yaxis=dict(
        gridcolor=IRON,
        linecolor=STEEL,
        tickfont=dict(color=STEAM),
        title_font=dict(color=BRONZE),
    ),
    legend=dict(
        bgcolor="rgba(42,32,22,0.8)",
        bordercolor=BRASS,
        borderwidth=1,
        font=dict(color=CREAM),
    ),
    colorway=CHART_COLORS,
    hoverlabel=dict(
        bgcolor=DARK_WOOD,
        bordercolor=BRASS,
        font=dict(color=CREAM, family="Georgia, serif"),
    ),
)

STEAM_TEMPLATE = go.layout.Template(layout=_STEAM_LAYOUT)
pio.templates["steam_train"] = STEAM_TEMPLATE
pio.templates.default = "steam_train"


def apply_steam_style(fig: go.Figure) -> go.Figure:
    """Apply the steam train style to an existing Plotly figure in-place."""
    fig.update_layout(template=STEAM_TEMPLATE)
    return fig


# ---------------------------------------------------------------------------
# CSS injection (call once from app.py)
# ---------------------------------------------------------------------------
STEAM_CSS = """
<style>
/* ---- Import a serif font reminiscent of vintage signage ---- */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Lora:wght@400;500;600&display=swap');

/* ---- Root variables ---- */
:root {
    --coal: #1B1B1B;
    --dark-iron: #2A2A2A;
    --iron: #4A4A4A;
    --steel: #8B8B8B;
    --brass: #B87333;
    --copper: #CD7F32;
    --bronze: #D4A574;
    --gold: #DAA520;
    --ember: #E25822;
    --mahogany: #3C1414;
    --dark-wood: #2A2016;
    --wood: #5C3A1E;
    --cream: #E8DCC8;
    --parchment: #FFF8E7;
    --steam: #C0C0C0;
}

/* ---- Global ---- */
.stApp {
    background: linear-gradient(175deg, #1B1B1B 0%, #1F1810 50%, #1B1B1B 100%);
    font-family: 'Lora', Georgia, serif;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2A1A0E 0%, #1C1008 100%);
    border-right: 3px solid var(--brass);
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: var(--bronze) !important;
    font-family: 'Lora', Georgia, serif;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--brass) !important;
    font-family: 'Playfair Display', Georgia, serif;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
}

[data-testid="stSidebar"] hr {
    border-color: var(--brass);
    opacity: 0.4;
}

/* ---- Headers ---- */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: var(--brass) !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.4);
    letter-spacing: 0.5px;
}

h1::before {
    content: "\\2736 ";
    color: var(--copper);
    font-size: 0.8em;
}

/* ---- Body text ---- */
p, li, span, label, .stMarkdown {
    font-family: 'Lora', Georgia, serif;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, var(--brass) 0%, var(--copper) 100%);
    color: #1B1B1B !important;
    border: 2px solid var(--gold);
    border-radius: 6px;
    font-family: 'Playfair Display', Georgia, serif;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-size: 0.85em;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(184,115,51,0.3);
}

.stButton > button:hover {
    background: linear-gradient(135deg, var(--copper) 0%, var(--gold) 100%);
    border-color: var(--parchment);
    box-shadow: 0 4px 16px rgba(184,115,51,0.5);
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0px);
    box-shadow: 0 1px 4px rgba(184,115,51,0.3);
}

/* ---- Download / primary button overrides ---- */
.stDownloadButton > button,
button[kind="primary"] {
    background: linear-gradient(135deg, var(--ember) 0%, var(--brass) 100%) !important;
    border: 2px solid var(--gold) !important;
    color: var(--parchment) !important;
}

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #2A2016 0%, #1C1008 100%);
    border: 1px solid var(--brass);
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.4), inset 0 1px 0 rgba(184,115,51,0.1);
}

[data-testid="stMetricLabel"] {
    color: var(--bronze) !important;
    font-family: 'Lora', Georgia, serif;
    text-transform: uppercase;
    font-size: 0.75em;
    letter-spacing: 1px;
}

[data-testid="stMetricValue"] {
    color: var(--parchment) !important;
    font-family: 'Playfair Display', Georgia, serif;
    font-weight: 700;
}

/* ---- Expander ---- */
.streamlit-expanderHeader {
    background-color: var(--dark-wood) !important;
    border: 1px solid var(--brass);
    border-radius: 6px;
    color: var(--bronze) !important;
    font-family: 'Playfair Display', Georgia, serif;
}

[data-testid="stExpander"] {
    border: 1px solid rgba(184,115,51,0.3);
    border-radius: 8px;
    background-color: rgba(42,32,22,0.3);
}

/* ---- Selectbox, multiselect, inputs ---- */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: var(--dark-iron) !important;
    border: 1px solid var(--brass) !important;
    border-radius: 6px;
    color: var(--cream) !important;
    font-family: 'Lora', Georgia, serif;
}

/* ---- Slider ---- */
.stSlider > div > div > div {
    color: var(--brass);
}

/* ---- Radio buttons ---- */
.stRadio > div {
    color: var(--cream);
}

/* ---- Dataframes ---- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--brass);
    border-radius: 8px;
    overflow: hidden;
}

/* ---- Progress bar ---- */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--ember), var(--brass), var(--gold)) !important;
}

/* ---- Divider ---- */
hr {
    border-color: var(--brass) !important;
    opacity: 0.3;
}

/* ---- Success/Info/Warning/Error boxes ---- */
.stSuccess, [data-testid="stNotification"][data-type="success"] {
    background-color: rgba(92,58,30,0.3) !important;
    border-left-color: var(--brass) !important;
}

.stInfo, [data-testid="stNotification"][data-type="info"] {
    background-color: rgba(42,42,42,0.5) !important;
    border-left-color: var(--steel) !important;
}

.stWarning, [data-testid="stNotification"][data-type="warning"] {
    background-color: rgba(226,88,34,0.15) !important;
    border-left-color: var(--ember) !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid var(--brass);
}

.stTabs [data-baseweb="tab"] {
    color: var(--steam);
    font-family: 'Playfair Display', Georgia, serif;
}

.stTabs [aria-selected="true"] {
    color: var(--brass) !important;
    border-bottom-color: var(--brass) !important;
}

/* ---- Checkbox ---- */
.stCheckbox span {
    color: var(--cream);
}

/* ---- Scrollbar styling ---- */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
::-webkit-scrollbar-track {
    background: var(--coal);
}
::-webkit-scrollbar-thumb {
    background: var(--iron);
    border-radius: 5px;
    border: 2px solid var(--coal);
}
::-webkit-scrollbar-thumb:hover {
    background: var(--brass);
}

/* ---- Decorative top banner ---- */
.stApp > header {
    background: linear-gradient(90deg, #1B1B1B, #2A1A0E, #1B1B1B) !important;
    border-bottom: 2px solid var(--brass);
}

/* ---- Spinner ---- */
.stSpinner > div {
    border-top-color: var(--brass) !important;
}

/* ---- Link styling ---- */
a {
    color: var(--copper) !important;
    text-decoration: none;
}
a:hover {
    color: var(--gold) !important;
    text-decoration: underline;
}
</style>
"""

# Banner HTML shown at the top of the app
HEADER_BANNER = """
<div style="
    text-align: center;
    padding: 20px 0 10px 0;
    margin-bottom: 10px;
    border-bottom: 2px solid #B87333;
    background: linear-gradient(90deg, transparent, rgba(184,115,51,0.08), transparent);
">
    <span style="font-size: 2.5em; line-height: 1;">&#x1F682;</span>
    <h2 style="
        font-family: 'Playfair Display', Georgia, serif;
        color: #B87333 !important;
        margin: 4px 0 2px 0;
        font-size: 1.6em;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    ">ECONOMIC SIMULATION EXPRESS</h2>
    <p style="
        font-family: 'Lora', Georgia, serif;
        color: #A9A9A9;
        font-size: 0.85em;
        margin: 0;
        letter-spacing: 1px;
        font-style: italic;
    ">Correlation Analysis &amp; Data Exploration &mdash; Full Steam Ahead</p>
</div>
"""

SIDEBAR_HEADER = """
<div style="
    text-align: center;
    padding: 10px 0;
    margin-bottom: 8px;
">
    <span style="font-size: 2em;">&#x1F682;</span>
    <div style="
        font-family: 'Playfair Display', Georgia, serif;
        color: #B87333;
        font-size: 1.1em;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-top: 2px;
    ">ECON EXPRESS</div>
    <div style="
        font-family: 'Lora', Georgia, serif;
        color: #8B8B8B;
        font-size: 0.7em;
        letter-spacing: 1px;
    ">DISPATCH BOARD</div>
</div>
"""

SIDEBAR_FOOTER = """
<div style="
    margin-top: 20px;
    padding: 12px;
    background: rgba(42,32,22,0.5);
    border: 1px solid rgba(184,115,51,0.3);
    border-radius: 8px;
    font-family: 'Lora', Georgia, serif;
    color: #A9A9A9;
    font-size: 0.78em;
    line-height: 1.6;
">
    <strong style="color: #B87333;">Data Sources</strong><br>
    &#x2022; <a href="https://data.worldbank.org/" target="_blank">World Bank Open Data</a><br>
    &#x2022; <a href="https://finance.yahoo.com/" target="_blank">Yahoo Finance</a><br><br>
    <strong style="color: #B87333;">Analysis Methods</strong><br>
    Pearson, Spearman, Kendall, Partial Corr.,
    Mutual Info, RF, GB, Lasso, Elastic Net,
    PCA, Autoencoder, Granger Causality<br><br>
    <strong style="color: #B87333;">Cargo &amp; Sentiment</strong><br>
    Freight trends, ML growth drivers,
    news sentiment via RSS
</div>
"""
