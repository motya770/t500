"""Tests for data_sources/stock_data.py module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from data_sources.stock_data import (
    TICKER_CATEGORIES,
    STOCK_TICKERS,
    get_all_tickers,
    download_stock_data,
    compute_annual_returns,
    merge_stock_with_economic,
    save_stock_dataset,
    load_stock_dataset,
)


class TestTickerConstants:
    def test_stock_tickers_not_empty(self):
        assert len(STOCK_TICKERS) > 0

    def test_voo_in_tickers(self):
        assert "VOO" in STOCK_TICKERS

    def test_qqq_in_tickers(self):
        assert "QQQ" in STOCK_TICKERS

    def test_categories_not_empty(self):
        assert len(TICKER_CATEGORIES) > 0

    def test_all_categories_have_tickers(self):
        for cat, tickers in TICKER_CATEGORIES.items():
            assert len(tickers) > 0, f"Category '{cat}' is empty"

    def test_voo_qqq_in_us_broad_market(self):
        assert "VOO" in TICKER_CATEGORIES["US Broad Market"]
        assert "QQQ" in TICKER_CATEGORIES["US Broad Market"]


class TestGetAllTickers:
    def test_returns_dict(self):
        result = get_all_tickers()
        assert isinstance(result, dict)

    def test_contains_voo_and_qqq(self):
        result = get_all_tickers()
        assert "VOO" in result
        assert "QQQ" in result

    def test_total_count(self):
        result = get_all_tickers()
        total = sum(len(v) for v in TICKER_CATEGORIES.values())
        assert len(result) == total


class TestDownloadStockData:
    @patch("data_sources.stock_data.yf")
    def test_returns_dataframe(self, mock_yf):
        mock_ticker = MagicMock()
        hist_data = pd.DataFrame({
            "Open": [100.0, 101.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 100.0],
            "Close": [103.0, 104.0],
            "Volume": [1000000, 1100000],
        }, index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
        hist_data.index.name = "Date"
        mock_ticker.history.return_value = hist_data
        mock_yf.Ticker.return_value = mock_ticker

        result = download_stock_data(["VOO"], start_year=2020, end_year=2020)
        assert isinstance(result, pd.DataFrame)
        assert "ticker" in result.columns
        assert "close" in result.columns
        assert "year" in result.columns

    @patch("data_sources.stock_data.yf")
    def test_empty_history(self, mock_yf):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        result = download_stock_data(["FAKE"], start_year=2020, end_year=2020)
        assert result.empty

    @patch("data_sources.stock_data.yf")
    def test_multiple_tickers(self, mock_yf):
        hist_data = pd.DataFrame({
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [103.0],
            "Volume": [1000000],
        }, index=pd.to_datetime(["2020-01-01"]))
        hist_data.index.name = "Date"

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_data
        mock_yf.Ticker.return_value = mock_ticker

        result = download_stock_data(["VOO", "QQQ"], start_year=2020, end_year=2020)
        assert isinstance(result, pd.DataFrame)
        assert set(result["ticker"].unique()) == {"VOO", "QQQ"}

    @patch("data_sources.stock_data.yf")
    def test_api_error_raises_runtime(self, mock_yf):
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("Network error")
        mock_yf.Ticker.return_value = mock_ticker

        with pytest.raises(RuntimeError, match="Failed to download data"):
            download_stock_data(["VOO"], start_year=2020, end_year=2020)


@pytest.fixture
def stock_data_fixture():
    """Create sample stock data for testing."""
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    data = []
    for ticker in ["VOO", "QQQ"]:
        base_price = 300.0 if ticker == "VOO" else 250.0
        for i, date in enumerate(dates):
            data.append({
                "ticker": ticker,
                "date": date,
                "year": date.year,
                "month": date.month,
                "open": base_price + i,
                "high": base_price + i + 5,
                "low": base_price + i - 3,
                "close": base_price + i + 2,
                "volume": 1000000 + i * 10000,
            })
    return pd.DataFrame(data)


class TestComputeAnnualReturns:
    def test_returns_dataframe(self, stock_data_fixture):
        result = compute_annual_returns(stock_data_fixture)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, stock_data_fixture):
        result = compute_annual_returns(stock_data_fixture)
        expected = {"ticker", "year", "annual_return_pct", "avg_close", "total_volume", "volatility"}
        assert expected.issubset(result.columns)

    def test_both_tickers_present(self, stock_data_fixture):
        result = compute_annual_returns(stock_data_fixture)
        assert set(result["ticker"].unique()) == {"VOO", "QQQ"}

    def test_positive_returns_for_increasing_prices(self, stock_data_fixture):
        result = compute_annual_returns(stock_data_fixture)
        # Prices are monotonically increasing, so returns should be positive
        assert (result["annual_return_pct"] > 0).all()

    def test_empty_input(self):
        result = compute_annual_returns(pd.DataFrame())
        assert result.empty

    def test_volume_is_positive(self, stock_data_fixture):
        result = compute_annual_returns(stock_data_fixture)
        assert (result["total_volume"] > 0).all()

    def test_volatility_non_negative(self, stock_data_fixture):
        result = compute_annual_returns(stock_data_fixture)
        assert (result["volatility"] >= 0).all()


class TestMergeStockWithEconomic:
    def test_merge_returns_dataframe(self, stock_data_fixture):
        annual = compute_annual_returns(stock_data_fixture)
        econ = pd.DataFrame({
            "country": ["USA", "USA", "GBR", "GBR"],
            "year": [2020, 2021, 2020, 2021],
            "gdp": [21000, 22000, 3000, 3100],
            "inflation": [1.2, 5.4, 0.9, 2.1],
        })

        result = merge_stock_with_economic(annual, econ, country="USA")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_merge_filters_by_country(self, stock_data_fixture):
        annual = compute_annual_returns(stock_data_fixture)
        econ = pd.DataFrame({
            "country": ["USA", "USA", "GBR"],
            "year": [2020, 2021, 2020],
            "gdp": [21000, 22000, 3000],
        })

        result = merge_stock_with_economic(annual, econ, country="USA")
        assert "country" in result.columns
        assert (result["country"] == "USA").all()

    def test_merge_contains_stock_columns(self, stock_data_fixture):
        annual = compute_annual_returns(stock_data_fixture)
        econ = pd.DataFrame({
            "country": ["USA", "USA"],
            "year": [2020, 2021],
            "gdp": [21000, 22000],
        })

        result = merge_stock_with_economic(annual, econ, country="USA")
        # Should contain pivoted stock columns
        stock_cols = [c for c in result.columns if "VOO" in c or "QQQ" in c]
        assert len(stock_cols) > 0

    def test_merge_empty_stock(self):
        econ = pd.DataFrame({"country": ["USA"], "year": [2020], "gdp": [21000]})
        result = merge_stock_with_economic(pd.DataFrame(), econ)
        assert result.empty

    def test_merge_empty_economic(self, stock_data_fixture):
        annual = compute_annual_returns(stock_data_fixture)
        result = merge_stock_with_economic(annual, pd.DataFrame())
        assert result.empty

    def test_merge_no_overlapping_years(self, stock_data_fixture):
        annual = compute_annual_returns(stock_data_fixture)
        econ = pd.DataFrame({
            "country": ["USA"],
            "year": [1990],
            "gdp": [10000],
        })
        result = merge_stock_with_economic(annual, econ, country="USA")
        assert result.empty

    def test_merge_nonexistent_country(self, stock_data_fixture):
        annual = compute_annual_returns(stock_data_fixture)
        econ = pd.DataFrame({
            "country": ["GBR"],
            "year": [2020],
            "gdp": [3000],
        })
        result = merge_stock_with_economic(annual, econ, country="USA")
        assert result.empty


class TestStockDataPersistence:
    def test_save_and_load(self, tmp_path):
        df = pd.DataFrame({
            "ticker": ["VOO"],
            "date": pd.to_datetime(["2020-01-01"]),
            "year": [2020],
            "month": [1],
            "open": [295.0],
            "high": [310.0],
            "low": [290.0],
            "close": [300.0],
            "volume": [1000000],
        })

        with patch("data_sources.database.DB_PATH", tmp_path / "test.db"), \
             patch("data_sources.database.DB_DIR", tmp_path), \
             patch("data_sources.stock_data.DATA_DIR", tmp_path):
            from data_sources.database import init_db
            init_db()
            save_stock_dataset(df, "test_stock")
            loaded = load_stock_dataset("test_stock")

        assert len(loaded) == len(df)
        assert "ticker" in loaded.columns
        assert loaded["close"].iloc[0] == pytest.approx(300.0)

    def test_load_nonexistent_raises(self, tmp_path):
        with patch("data_sources.database.DB_PATH", tmp_path / "test.db"), \
             patch("data_sources.database.DB_DIR", tmp_path), \
             patch("data_sources.stock_data.DATA_DIR", tmp_path):
            from data_sources.database import init_db
            init_db()
            with pytest.raises(FileNotFoundError):
                load_stock_dataset("nonexistent")

    def test_save_creates_directory(self, tmp_path):
        new_dir = tmp_path / "new_subdir"
        df = pd.DataFrame({
            "ticker": ["VOO"],
            "date": pd.to_datetime(["2020-01-01"]),
            "year": [2020],
            "month": [1],
            "close": [300.0],
        })

        with patch("data_sources.database.DB_PATH", new_dir / "test.db"), \
             patch("data_sources.database.DB_DIR", new_dir), \
             patch("data_sources.stock_data.DATA_DIR", new_dir):
            from data_sources.database import init_db
            init_db()
            save_stock_dataset(df, "test")

        assert new_dir.exists()

    def test_csv_fallback_on_load(self, tmp_path):
        """Legacy CSV files should still be loadable."""
        df = pd.DataFrame({
            "ticker": ["VOO"],
            "date": ["2020-01-01"],
            "close": [300.0],
        })
        df.to_csv(tmp_path / "legacy_stock.csv", index=False)

        with patch("data_sources.database.DB_PATH", tmp_path / "test.db"), \
             patch("data_sources.database.DB_DIR", tmp_path), \
             patch("data_sources.stock_data.DATA_DIR", tmp_path):
            from data_sources.database import init_db
            init_db()
            loaded = load_stock_dataset("legacy_stock")

        assert len(loaded) == 1
        assert loaded["ticker"].iloc[0] == "VOO"
