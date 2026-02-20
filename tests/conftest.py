"""Shared test fixtures for all test modules."""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def economic_df():
    """Create a sample economic DataFrame with country/year and indicator columns.

    Mimics the wide-format DataFrame produced by download_multiple_indicators().
    """
    np.random.seed(42)
    countries = ["USA", "GBR", "DEU"]
    years = list(range(2018, 2024))
    rows = []
    for country in countries:
        for year in years:
            rows.append({
                "country": country,
                "year": year,
                "NY.GDP.MKTP.CD": np.random.uniform(1e12, 5e12),
                "FP.CPI.TOTL.ZG": np.random.uniform(0, 10),
                "SL.UEM.TOTL.ZS": np.random.uniform(2, 15),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def indicator_names():
    """Sample indicator_names dict for session state."""
    return {
        "NY.GDP.MKTP.CD": "GDP (current US$)",
        "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
        "SL.UEM.TOTL.ZS": "Unemployment, total (% of total labor force)",
    }


@pytest.fixture
def numeric_df():
    """Pure numeric DataFrame (no country/year) for analysis tests."""
    np.random.seed(42)
    n = 50
    x = np.random.randn(n)
    y = x * 2 + np.random.randn(n) * 0.5
    z = np.random.randn(n)
    return pd.DataFrame({"ind_a": x, "ind_b": y, "ind_c": z})
