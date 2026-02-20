# :steam_locomotive: Econ Express — Steam Train Dashboard

An interactive economic simulation and correlation analysis dashboard built with Streamlit. Fetches real-world data from the World Bank API, Yahoo Finance, and news feeds, then provides rich visualizations and advanced statistical/ML-based analysis.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## :sparkles: Features

### :package: Data Download
- **World Bank Indicators** — 60+ economic indicators across 10 categories (GDP, Trade, Inflation, Employment, Government, Population, Financial, Education, Energy, Health)
- **Stock / ETF Data** — Fetch historical market data via Yahoo Finance
- **News Sentiment** — RSS news feeds with sentiment analysis
- Predefined country groups (G7, BRICS, EU, East Asia, Latin America, Middle East)
- Save and reload datasets as CSV

### :mag: Explore & Visualize
Six interactive chart types powered by Plotly:
- :chart_with_upwards_trend: Time Series
- :bar_chart: Country Comparison
- :small_red_triangle: Scatter Plot
- :bell: Distribution
- :fire: Heatmap
- :package: Box Plot

### :link: Correlation Analysis
13 analysis methods across three tiers:

| Tier | Methods |
|------|---------|
| :triangular_ruler: **Classical Statistics** | Pearson, Spearman, Kendall, Partial Correlation, Correlation with p-values |
| :gear: **ML / Information Theory** | Mutual Information, Feature Importance (RF, GB, Lasso, Elastic Net), Cross-Validated Scores, PCA |
| :brain: **Deep Learning / Time-Series** | Autoencoder (PyTorch), Granger Causality |

### :chart_with_downwards_trend: Inflation-Stock Models
Statistical modeling of relationships between inflation indicators and stock/ETF performance.

### :airplane: Cargo Plane Analysis
Air freight and cargo transport data analysis with economic correlation.

### :oil_drum: Oil Tanker Analysis
Maritime oil transport data exploration and economic impact modeling.

### :newspaper: News Sentiment
RSS feed aggregation with TextBlob-powered sentiment scoring tied to economic indicators.

---

## :rocket: Quick Start

### Prerequisites
- Python 3.11+
- Internet connection (for World Bank API, Yahoo Finance, news feeds)

### Installation

```bash
# Clone the repository
git clone https://github.com/motya770/t500.git
cd t500

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
# Option 1: via entry point
python main.py

# Option 2: direct Streamlit launch
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## :building_construction: Architecture

```
World Bank API / Yahoo Finance / RSS Feeds
    |
    v
data_sources/
    world_bank.py        --> downloads indicators --> saves CSV to data/
    stock_data.py         --> fetches stock/ETF prices
    oil_data.py           --> oil tanker & maritime data
    news_fetcher.py       --> RSS feed aggregation
    |
    v   st.session_state
ui/
    page_download.py      --> indicator/country/year selection
    page_stock_download.py --> stock & ETF data fetching
    page_explore.py       --> 6 interactive visualization types
    page_correlations.py  --> runs 13 analysis methods
    page_inflation_stock.py --> inflation-stock modeling
    page_cargo.py         --> cargo plane analysis
    page_oil_tankers.py   --> oil tanker analysis
    page_news_sentiment.py --> news sentiment dashboard
    theme.py              --> steam train visual theme
    |
    v
analysis/
    correlations.py       --> statistical & ML correlation methods
    inflation_stock_models.py --> inflation-stock regression
    cargo_analysis.py     --> air freight analytics
    oil_analysis.py       --> oil transport analytics
    sentiment.py          --> text sentiment scoring
```

---

## :file_folder: Project Structure

```
.
├── main.py                      # Entry point (launches Streamlit)
├── app.py                       # App config, theme injection, page routing
├── requirements.txt             # Python dependencies
├── .github/workflows/ci.yml     # GitHub Actions CI (pytest + coverage)
├── analysis/
│   ├── correlations.py          # 13 statistical/ML correlation methods
│   ├── inflation_stock_models.py
│   ├── cargo_analysis.py
│   ├── oil_analysis.py
│   └── sentiment.py
├── data_sources/
│   ├── world_bank.py            # World Bank API client + CSV persistence
│   ├── stock_data.py            # Yahoo Finance data fetcher
│   ├── oil_data.py              # Oil/maritime data source
│   └── news_fetcher.py          # RSS news feed fetcher
├── ui/
│   ├── page_download.py         # World Bank data download page
│   ├── page_stock_download.py   # Stock/ETF download page
│   ├── page_explore.py          # Data exploration & visualization
│   ├── page_correlations.py     # Correlation analysis dashboard
│   ├── page_inflation_stock.py  # Inflation-stock modeling page
│   ├── page_cargo.py            # Cargo plane analysis page
│   ├── page_oil_tankers.py      # Oil tanker analysis page
│   ├── page_news_sentiment.py   # News sentiment page
│   └── theme.py                 # Steam train CSS theme & Plotly template
├── tests/                       # pytest test suite
│   ├── conftest.py
│   ├── test_correlations.py
│   ├── test_world_bank.py
│   ├── test_page_*.py
│   └── ...
└── data/                        # Runtime CSV storage (gitignored)
```

---

## :test_tube: Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests with coverage
pytest --cov=analysis --cov=data_sources --cov=ui --cov-report=term-missing -v
```

CI runs automatically on push and pull requests to `master` via GitHub Actions.

---

## :wrench: Key Dependencies

| Package | Purpose |
|---------|---------|
| :red_circle: `streamlit` | Web UI framework |
| :panda_face: `pandas` / `numpy` | Data manipulation |
| :gear: `scikit-learn` | ML models (RF, GB, Lasso, Elastic Net, PCA) |
| :bar_chart: `scipy` | Statistical tests & correlations |
| :fire: `torch` | Deep learning (autoencoder) |
| :globe_with_meridians: `wbgapi` | World Bank API client |
| :chart_with_upwards_trend: `plotly` | Interactive charts |
| :art: `seaborn` / `matplotlib` | Static plotting |
| :money_with_wings: `yfinance` | Stock & ETF market data |
| :newspaper: `feedparser` | RSS feed parsing |
| :speech_balloon: `textblob` | Sentiment analysis |

---

## :handshake: Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and add tests
4. Run the test suite (`pytest -v`)
5. Commit and push (`git push origin feature/my-feature`)
6. Open a Pull Request

---

## :scroll: License

This project is open source. See the repository for license details.
