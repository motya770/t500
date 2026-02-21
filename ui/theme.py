"""Futuristic Blue theme for the Economic Simulation dashboard.

Provides a consistent color palette, CSS injection, and Plotly chart styling
with a modern, futuristic aesthetic: deep navy backgrounds, electric blue accents,
cyan highlights, and clean sans-serif typography.
"""

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Color palette  (legacy names kept for backward-compat across all pages)
# ---------------------------------------------------------------------------
# Backgrounds
COAL = "#060B18"         # deep space black
DARK_IRON = "#0B1224"    # dark navy
IRON = "#162040"         # medium navy
STEEL = "#5B7BA0"        # steel blue

# Dark accent surfaces
MAHOGANY = "#081428"     # deep blue
DARK_WOOD = "#0A1830"    # dark blue
WOOD = "#1B3358"         # mid blue

# Primary accent colors
BRASS = "#00B4D8"        # electric cyan (primary accent)
COPPER = "#38BDF8"       # bright sky blue
BRONZE = "#7DD3FC"       # light blue
GOLD = "#06D6A0"         # neon teal-green
EMBER = "#818CF8"        # indigo / purple
FIRE_ORANGE = "#22D3EE"  # bright cyan

# Text / neutrals
CREAM = "#E2E8F0"        # silver-white
PARCHMENT = "#F1F5F9"    # near-white
STEAM = "#94A3B8"        # slate
SMOKE = "#64748B"        # muted slate

# Plotly chart color sequence (futuristic spectrum)
CHART_COLORS = [
    "#00B4D8",  # electric cyan
    "#818CF8",  # indigo
    "#06D6A0",  # emerald
    "#38BDF8",  # sky blue
    "#A78BFA",  # violet
    "#22D3EE",  # bright cyan
    "#2DD4BF",  # teal
    "#F472B6",  # pink
    "#67E8F9",  # light cyan
    "#34D399",  # green
    "#94A3B8",  # slate
    "#C084FC",  # purple
]

# Heatmap color scales
HEATMAP_SCALE = [
    [0.0, "#060B18"],
    [0.15, "#0B1E42"],
    [0.3, "#0E3B6E"],
    [0.5, "#00B4D8"],
    [0.7, "#38BDF8"],
    [0.85, "#7DD3FC"],
    [1.0, "#F1F5F9"],
]

DIVERGING_SCALE = [
    [0.0, "#7C3AED"],
    [0.25, "#818CF8"],
    [0.5, "#1E293B"],
    [0.75, "#06B6D4"],
    [1.0, "#06D6A0"],
]

# ---------------------------------------------------------------------------
# Plotly layout template
# ---------------------------------------------------------------------------
_THEME_LAYOUT = go.Layout(
    paper_bgcolor=COAL,
    plot_bgcolor=DARK_IRON,
    font=dict(family="Inter, Segoe UI, Roboto, sans-serif", color=CREAM, size=13),
    title=dict(font=dict(family="Inter, Segoe UI, Roboto, sans-serif", color=BRASS, size=18)),
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
        bgcolor="rgba(11,18,36,0.85)",
        bordercolor=BRASS,
        borderwidth=1,
        font=dict(color=CREAM),
    ),
    colorway=CHART_COLORS,
    hoverlabel=dict(
        bgcolor=DARK_WOOD,
        bordercolor=BRASS,
        font=dict(color=CREAM, family="Inter, Segoe UI, Roboto, sans-serif"),
    ),
)

STEAM_TEMPLATE = go.layout.Template(layout=_THEME_LAYOUT)
pio.templates["steam_train"] = STEAM_TEMPLATE
pio.templates.default = "steam_train"


def apply_steam_style(fig: go.Figure) -> go.Figure:
    """Apply the futuristic blue style to an existing Plotly figure in-place."""
    fig.update_layout(template=STEAM_TEMPLATE)
    return fig


# ---------------------------------------------------------------------------
# CSS injection (call once from app.py)
# ---------------------------------------------------------------------------
STEAM_CSS = """
<style>
/* ---- Import modern sans-serif font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---- Root variables ---- */
:root {
    --deep-space: #060B18;
    --dark-navy: #0B1224;
    --navy: #162040;
    --steel-blue: #5B7BA0;
    --electric-cyan: #00B4D8;
    --sky-blue: #38BDF8;
    --light-blue: #7DD3FC;
    --neon-teal: #06D6A0;
    --indigo: #818CF8;
    --bright-cyan: #22D3EE;
    --silver-white: #E2E8F0;
    --near-white: #F1F5F9;
    --slate: #94A3B8;
    --dark-blue: #0A1830;
    --mid-blue: #1B3358;

    /* Legacy aliases */
    --coal: var(--deep-space);
    --brass: var(--electric-cyan);
    --copper: var(--sky-blue);
    --cream: var(--silver-white);
    --ember: var(--indigo);
    --steam: var(--slate);
}

/* ---- Global ---- */
.stApp {
    background:
        radial-gradient(ellipse at 20% 50%, rgba(0,180,216,0.03) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(129,140,248,0.03) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(6,214,160,0.02) 0%, transparent 50%),
        linear-gradient(175deg, #060B18 0%, #0B1224 40%, #081020 100%);
    font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1224 0%, #060B18 100%);
    border-right: 1px solid rgba(0,180,216,0.2);
    box-shadow: 2px 0 20px rgba(0,180,216,0.05);
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: var(--slate) !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--electric-cyan) !important;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(0,180,216,0.15);
}

/* ---- Headers ---- */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Inter', sans-serif !important;
    color: var(--electric-cyan) !important;
    font-weight: 600;
    letter-spacing: -0.01em;
}

h1 {
    background: linear-gradient(135deg, #00B4D8, #38BDF8, #818CF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ---- Body text ---- */
p, li, span, label, .stMarkdown {
    font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,180,216,0.15) 0%, rgba(129,140,248,0.15) 100%);
    color: var(--electric-cyan) !important;
    border: 1px solid rgba(0,180,216,0.4);
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    letter-spacing: 0.02em;
    font-size: 0.85em;
    transition: all 0.3s ease;
    box-shadow: 0 0 15px rgba(0,180,216,0.08);
    backdrop-filter: blur(10px);
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,180,216,0.25) 0%, rgba(129,140,248,0.25) 100%);
    border-color: var(--electric-cyan);
    box-shadow: 0 0 25px rgba(0,180,216,0.2), 0 0 50px rgba(0,180,216,0.05);
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0px);
    box-shadow: 0 0 10px rgba(0,180,216,0.15);
}

/* ---- Primary button ---- */
.stDownloadButton > button,
button[kind="primary"] {
    background: linear-gradient(135deg, #00B4D8 0%, #818CF8 100%) !important;
    border: 1px solid rgba(56,189,248,0.5) !important;
    color: var(--near-white) !important;
    box-shadow: 0 0 20px rgba(0,180,216,0.2);
}

.stDownloadButton > button:hover,
button[kind="primary"]:hover {
    box-shadow: 0 0 30px rgba(0,180,216,0.35), 0 0 60px rgba(0,180,216,0.1) !important;
}

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(11,18,36,0.8) 0%, rgba(22,32,64,0.6) 100%);
    border: 1px solid rgba(0,180,216,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(0,180,216,0.05);
    backdrop-filter: blur(10px);
}

[data-testid="stMetricLabel"] {
    color: var(--slate) !important;
    font-family: 'Inter', sans-serif;
    text-transform: uppercase;
    font-size: 0.72em;
    letter-spacing: 1.5px;
    font-weight: 500;
}

[data-testid="stMetricValue"] {
    color: var(--near-white) !important;
    font-family: 'JetBrains Mono', 'Inter', monospace;
    font-weight: 600;
}

/* ---- Expander ---- */
.streamlit-expanderHeader {
    background-color: rgba(11,18,36,0.6) !important;
    border: 1px solid rgba(0,180,216,0.15);
    border-radius: 8px;
    color: var(--sky-blue) !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stExpander"] {
    border: 1px solid rgba(0,180,216,0.1);
    border-radius: 10px;
    background-color: rgba(11,18,36,0.3);
}

/* ---- Selectbox, multiselect, inputs ---- */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: rgba(11,18,36,0.8) !important;
    border: 1px solid rgba(0,180,216,0.25) !important;
    border-radius: 8px;
    color: var(--silver-white) !important;
    font-family: 'Inter', sans-serif;
    backdrop-filter: blur(10px);
}

.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--electric-cyan) !important;
    box-shadow: 0 0 12px rgba(0,180,216,0.15) !important;
}

/* ---- Slider ---- */
.stSlider > div > div > div {
    color: var(--electric-cyan);
}

/* ---- Dataframes ---- */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,180,216,0.2);
    border-radius: 10px;
    overflow: hidden;
}

/* ---- Progress bar ---- */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00B4D8, #818CF8, #06D6A0) !important;
}

/* ---- Divider ---- */
hr {
    border-color: rgba(0,180,216,0.15) !important;
}

/* ---- Success/Info/Warning/Error boxes ---- */
.stSuccess, [data-testid="stNotification"][data-type="success"] {
    background-color: rgba(6,214,160,0.1) !important;
    border-left-color: var(--neon-teal) !important;
}

.stInfo, [data-testid="stNotification"][data-type="info"] {
    background-color: rgba(0,180,216,0.1) !important;
    border-left-color: var(--electric-cyan) !important;
}

.stWarning, [data-testid="stNotification"][data-type="warning"] {
    background-color: rgba(251,191,36,0.1) !important;
    border-left-color: #FBBF24 !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid rgba(0,180,216,0.2);
}

.stTabs [data-baseweb="tab"] {
    color: var(--slate);
    font-family: 'Inter', sans-serif;
}

.stTabs [aria-selected="true"] {
    color: var(--electric-cyan) !important;
    border-bottom-color: var(--electric-cyan) !important;
}

/* ---- Checkbox ---- */
.stCheckbox span {
    color: var(--silver-white);
}

/* ---- Scrollbar styling ---- */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: var(--deep-space);
}
::-webkit-scrollbar-thumb {
    background: var(--navy);
    border-radius: 4px;
    border: 2px solid var(--deep-space);
}
::-webkit-scrollbar-thumb:hover {
    background: var(--electric-cyan);
}

/* ---- Top header bar ---- */
.stApp > header {
    background: linear-gradient(90deg, #060B18, #0B1224, #060B18) !important;
    border-bottom: 1px solid rgba(0,180,216,0.15);
}

/* ---- Spinner ---- */
.stSpinner > div {
    border-top-color: var(--electric-cyan) !important;
}

/* ---- Link styling ---- */
a {
    color: var(--sky-blue) !important;
    text-decoration: none;
}
a:hover {
    color: var(--electric-cyan) !important;
    text-decoration: none;
    text-shadow: 0 0 8px rgba(0,180,216,0.3);
}

/* ---- Animated glow line under header banner ---- */
@keyframes glowPulse {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 1; }
}
</style>
"""

# Banner HTML shown at the top of the app
HEADER_BANNER = """
<div style="
    text-align: center;
    padding: 28px 0 18px 0;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(0,180,216,0.2);
    background: radial-gradient(ellipse at 50% 100%, rgba(0,180,216,0.06) 0%, transparent 70%);
    position: relative;
">
    <div style="
        font-size: 2em;
        line-height: 1;
        margin-bottom: 8px;
        filter: drop-shadow(0 0 8px rgba(0,180,216,0.4));
    ">&#x26A1;</div>
    <h2 style="
        font-family: 'Inter', sans-serif;
        color: #00B4D8 !important;
        margin: 4px 0 6px 0;
        font-size: 1.5em;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        background: linear-gradient(135deg, #00B4D8, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    ">ECONOMIC SIMULATION</h2>
    <p style="
        font-family: 'Inter', sans-serif;
        color: #64748B;
        font-size: 0.8em;
        margin: 0;
        letter-spacing: 2px;
        font-weight: 400;
        text-transform: uppercase;
    ">Correlation Analysis &amp; Data Intelligence</p>
    <div style="
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, #00B4D8, #818CF8);
        margin: 12px auto 0;
        border-radius: 1px;
    "></div>
</div>
"""

SIDEBAR_HEADER = """
<div style="
    text-align: center;
    padding: 12px 0;
    margin-bottom: 8px;
">
    <div style="
        font-size: 1.8em;
        filter: drop-shadow(0 0 6px rgba(0,180,216,0.3));
    ">&#x26A1;</div>
    <div style="
        font-family: 'Inter', sans-serif;
        color: #00B4D8;
        font-size: 1em;
        font-weight: 700;
        letter-spacing: 2px;
        margin-top: 4px;
        text-transform: uppercase;
    ">ECON SIM</div>
    <div style="
        font-family: 'Inter', sans-serif;
        color: #64748B;
        font-size: 0.65em;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 500;
    ">CONTROL PANEL</div>
</div>
"""

SIDEBAR_FOOTER = """
<div style="
    margin-top: 20px;
    padding: 14px;
    background: rgba(11,18,36,0.6);
    border: 1px solid rgba(0,180,216,0.1);
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    color: #64748B;
    font-size: 0.75em;
    line-height: 1.7;
    backdrop-filter: blur(10px);
">
    <strong style="color: #00B4D8; font-weight: 600; letter-spacing: 0.5px;">Data Sources</strong><br>
    &#x2022; <a href="https://data.worldbank.org/" target="_blank" style="color: #38BDF8 !important;">World Bank Open Data</a><br>
    &#x2022; <a href="https://finance.yahoo.com/" target="_blank" style="color: #38BDF8 !important;">Yahoo Finance</a><br><br>
    <strong style="color: #00B4D8; font-weight: 600; letter-spacing: 0.5px;">Analysis Methods</strong><br>
    Pearson, Spearman, Kendall, Partial Corr.,
    Mutual Info, RF, GB, Lasso, Elastic Net,
    PCA, Autoencoder, Granger Causality<br><br>
    <strong style="color: #00B4D8; font-weight: 600; letter-spacing: 0.5px;">Cargo &amp; Sentiment</strong><br>
    Freight trends, ML growth drivers,
    news sentiment via RSS
</div>
"""
