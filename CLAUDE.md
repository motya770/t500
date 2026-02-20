# CLAUDE.md

## Project Overview

Economic Simulation & Correlation Analysis Dashboard — a Streamlit web application that fetches economic indicator data from the World Bank API and provides interactive visualization and advanced statistical/ML-based correlation analysis.

## Repository Structure

```
├── main.py                  # Entry point (launches Streamlit via subprocess)
├── app.py                   # Streamlit app config and page routing
├── requirements.txt         # Python dependencies
├── .gitignore
├── analysis/
│   ├── __init__.py
│   └── correlations.py      # Statistical and ML correlation methods
├── data_sources/
│   ├── __init__.py
│   └── world_bank.py        # World Bank API data fetching and CSV persistence
└── ui/
    ├── __init__.py
    ├── page_download.py      # Data download page
    ├── page_explore.py       # Data exploration & visualization page
    └── page_correlations.py  # Correlation analysis page
```

**Runtime data**: Downloaded datasets are saved as CSV files in `data/` (gitignored).

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Option 1: via entry point
python main.py

# Option 2: direct Streamlit launch
streamlit run app.py
```

Requires an internet connection for World Bank API access.

## Key Dependencies

| Package | Purpose |
|---------|---------|
| streamlit (>=1.30.0) | Web UI framework |
| pandas (>=2.0.0), numpy (>=1.24.0) | Data manipulation |
| scikit-learn (>=1.3.0) | ML models (RF, GB, Lasso, Elastic Net) |
| scipy (>=1.11.0) | Statistical tests |
| torch (>=2.0.0) | Deep learning (autoencoder) |
| wbgapi (>=1.0.0) | World Bank API client |
| plotly (>=5.18.0) | Interactive charts |
| seaborn (>=0.13.0), matplotlib (>=3.8.0) | Static plotting |

## Architecture & Data Flow

```
World Bank API
    ↓  wbgapi
data_sources/world_bank.py       → downloads indicators → saves CSV to data/
    ↓
ui/page_download.py              → user selects indicators/countries/years
    ↓  st.session_state
ui/page_explore.py               → 6 visualization types (Plotly)
ui/page_correlations.py          → calls analysis/correlations.py
    ↓
analysis/correlations.py         → returns matrices, scores, p-values
```

- **Session state** (`st.session_state`) is the shared data bus between pages. `page_download.py` stores `current_dataset` and `indicator_names`; the other pages read them or load saved datasets from disk.
- `correlations.py` is stateless — pure functions only, called from `page_correlations.py`.
- `world_bank.py` handles all external I/O (API calls) and file persistence.

## Module Details

### `data_sources/world_bank.py`
- `INDICATOR_CATEGORIES`: dict of 60+ World Bank indicators across 10 categories (GDP, Trade, Inflation, Employment, Government, Population, Financial, Education, Energy, Health).
- `download_indicator()` / `download_multiple_indicators()`: fetch data via `wbgapi`, return long-format or merged wide-format DataFrames.
- `save_dataset()` / `load_dataset()` / `list_saved_datasets()`: CSV persistence in `data/`.
- `get_country_groups()`: predefined groupings (G7, BRICS, EU, East Asia, Latin America, Middle East).

### `analysis/correlations.py`
13 analysis methods in three tiers:

**Classical statistics**: `pearson_correlation`, `spearman_correlation`, `kendall_correlation`, `partial_correlation`, `correlation_with_pvalues`.

**ML / Information theory**: `mutual_information_matrix`, `ml_feature_importance_matrix` (RF, GB, Lasso, Elastic Net), `ml_cross_validated_scores`, `pca_analysis`.

**Deep learning / Time-series**: `CorrelationAutoencoder` (PyTorch nn.Module) + `train_autoencoder`, `granger_causality_test`.

Utility: `get_top_correlations` extracts top-N pairs from any correlation matrix.

### `ui/` pages
- **page_download.py**: indicator/country/year selection, download with progress bar, save to CSV.
- **page_explore.py**: 6 chart types — time series, country comparison, scatter, distribution, heatmap, box plot.
- **page_correlations.py**: data preparation (aggregation mode, missing-value handling), runs any of the 13 analysis methods, renders heatmaps and result tables.

## Conventions

- **No test suite**: there are no tests currently. If adding tests, use `pytest` and place them in a `tests/` directory.
- **No linter/formatter config**: no `.flake8`, `pyproject.toml`, or similar. Follow PEP 8 style consistent with existing code.
- **Module init files**: all `__init__.py` files are empty. Imports use explicit paths (`from ui.page_download import render`).
- **Page pattern**: each UI page exports a single `render()` function called from `app.py`.
- **Lazy imports**: `app.py` imports page modules inside conditionals to avoid loading unused pages.
- **DataFrames**: indicator data uses long format (`country`, `year`, indicator columns) after merging. The columns `country` and `year` are metadata; all other columns are indicator values.
- **Indicator naming**: indicator codes (e.g., `NY.GDP.MKTP.CD`) are used as column names. Human-readable labels come from `INDICATOR_CATEGORIES` via helper functions.

## Common Tasks

**Adding a new analysis method**:
1. Implement the function in `analysis/correlations.py` (stateless, takes a DataFrame, returns results).
2. Add a rendering function in `ui/page_correlations.py` (prefix with `_run_`).
3. Add the method name to the multiselect list in `page_correlations.py`'s `render()`.

**Adding a new visualization type**:
1. Add the chart type string to the selectbox in `ui/page_explore.py`.
2. Add a corresponding `elif` block with Plotly chart code.

**Adding new World Bank indicators**:
1. Add entries to the appropriate category in `INDICATOR_CATEGORIES` in `data_sources/world_bank.py`.

**Adding a new data source**:
1. Create a new module in `data_sources/` following the pattern of `world_bank.py`.
2. Wire it into a download page or extend `page_download.py`.
