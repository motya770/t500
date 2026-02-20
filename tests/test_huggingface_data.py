"""Tests for data_sources/huggingface_data.py."""

import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from data_sources.huggingface_data import (
    is_cache_available,
    get_cache_info,
    get_hf_indicator_catalog,
    download_from_hf,
    download_hf_dataset,
    _build_catalog,
    CACHE_DIR,
    CACHE_FILE,
    CATALOG_FILE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_long_df():
    """Sample long-format DataFrame matching HuggingFace WDI structure."""
    return pd.DataFrame({
        "country_name": ["United States"] * 4 + ["United Kingdom"] * 4,
        "country_code": ["USA"] * 4 + ["GBR"] * 4,
        "indicator_name": ["GDP (current US$)", "Inflation (%)"] * 4,
        "indicator_code": ["NY.GDP.MKTP.CD", "FP.CPI.TOTL.ZG"] * 4,
        "year": [2019, 2019, 2020, 2020, 2019, 2019, 2020, 2020],
        "indicator_value": [21.4e12, 1.8, 20.9e12, 1.2, 2.8e12, 1.7, 2.7e12, 0.9],
    })


@pytest.fixture
def cache_dir(tmp_path):
    """Set up a temporary cache directory with test data."""
    hf_cache = tmp_path / "hf_cache"
    hf_cache.mkdir()
    return hf_cache


@pytest.fixture
def cached_parquet(cache_dir, sample_long_df):
    """Write sample data as a Parquet file in the temp cache dir."""
    path = cache_dir / "wdi.parquet"
    sample_long_df.to_parquet(path, index=False)
    return path


@pytest.fixture
def cached_catalog(cache_dir):
    """Write a sample indicator catalog JSON."""
    catalog = {
        "NY.GDP.MKTP.CD": "GDP (current US$)",
        "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    }
    path = cache_dir / "indicator_catalog.json"
    path.write_text(json.dumps(catalog))
    return path


# ---------------------------------------------------------------------------
# is_cache_available
# ---------------------------------------------------------------------------

class TestIsCacheAvailable:
    def test_returns_false_when_missing(self, tmp_path):
        with patch("data_sources.huggingface_data.CACHE_FILE", tmp_path / "missing.parquet"):
            assert is_cache_available() is False

    def test_returns_true_when_present(self, cached_parquet):
        with patch("data_sources.huggingface_data.CACHE_FILE", cached_parquet):
            assert is_cache_available() is True


# ---------------------------------------------------------------------------
# get_cache_info
# ---------------------------------------------------------------------------

class TestGetCacheInfo:
    def test_unavailable_when_no_file(self, tmp_path):
        with patch("data_sources.huggingface_data.CACHE_FILE", tmp_path / "missing.parquet"):
            info = get_cache_info()
            assert info["available"] is False

    def test_available_with_size(self, cached_parquet, cached_catalog):
        with patch("data_sources.huggingface_data.CACHE_FILE", cached_parquet), \
             patch("data_sources.huggingface_data.CATALOG_FILE", cached_catalog):
            info = get_cache_info()
            assert info["available"] is True
            assert "size_mb" in info
            assert info["indicator_count"] == 2


# ---------------------------------------------------------------------------
# get_hf_indicator_catalog
# ---------------------------------------------------------------------------

class TestGetHfIndicatorCatalog:
    def test_loads_from_json(self, cached_catalog):
        with patch("data_sources.huggingface_data.CATALOG_FILE", cached_catalog), \
             patch("data_sources.huggingface_data.CACHE_FILE", Path("/nonexistent")):
            catalog = get_hf_indicator_catalog()
            assert "NY.GDP.MKTP.CD" in catalog
            assert len(catalog) == 2

    def test_returns_empty_when_no_cache(self, tmp_path):
        with patch("data_sources.huggingface_data.CATALOG_FILE", tmp_path / "missing.json"), \
             patch("data_sources.huggingface_data.CACHE_FILE", tmp_path / "missing.parquet"):
            catalog = get_hf_indicator_catalog()
            assert catalog == {}

    def test_rebuilds_from_parquet(self, cached_parquet, cache_dir):
        catalog_path = cache_dir / "indicator_catalog.json"
        # Ensure catalog doesn't exist yet
        if catalog_path.exists():
            catalog_path.unlink()

        with patch("data_sources.huggingface_data.CATALOG_FILE", catalog_path), \
             patch("data_sources.huggingface_data.CACHE_FILE", cached_parquet):
            catalog = get_hf_indicator_catalog()
            assert "NY.GDP.MKTP.CD" in catalog
            assert "FP.CPI.TOTL.ZG" in catalog
            # Catalog JSON should now exist
            assert catalog_path.exists()


# ---------------------------------------------------------------------------
# _build_catalog
# ---------------------------------------------------------------------------

class TestBuildCatalog:
    def test_writes_json(self, cache_dir, sample_long_df):
        catalog_path = cache_dir / "indicator_catalog.json"
        with patch("data_sources.huggingface_data.CATALOG_FILE", catalog_path):
            _build_catalog(sample_long_df)
            assert catalog_path.exists()
            data = json.loads(catalog_path.read_text())
            assert "NY.GDP.MKTP.CD" in data
            assert "FP.CPI.TOTL.ZG" in data


# ---------------------------------------------------------------------------
# download_from_hf
# ---------------------------------------------------------------------------

class TestDownloadFromHf:
    def test_returns_wide_dataframe(self, cached_parquet, cached_catalog):
        with patch("data_sources.huggingface_data.CACHE_FILE", cached_parquet), \
             patch("data_sources.huggingface_data.CATALOG_FILE", cached_catalog):
            df, failed = download_from_hf(
                ["NY.GDP.MKTP.CD", "FP.CPI.TOTL.ZG"],
                ["USA", "GBR"],
                2019, 2020,
            )

        assert isinstance(df, pd.DataFrame)
        assert "country" in df.columns
        assert "year" in df.columns
        assert "NY.GDP.MKTP.CD" in df.columns
        assert "FP.CPI.TOTL.ZG" in df.columns
        assert failed == []

    def test_filters_by_country(self, cached_parquet, cached_catalog):
        with patch("data_sources.huggingface_data.CACHE_FILE", cached_parquet), \
             patch("data_sources.huggingface_data.CATALOG_FILE", cached_catalog):
            df, _ = download_from_hf(
                ["NY.GDP.MKTP.CD"],
                ["USA"],
                2019, 2020,
            )

        assert set(df["country"].unique()) == {"USA"}

    def test_filters_by_year(self, cached_parquet, cached_catalog):
        with patch("data_sources.huggingface_data.CACHE_FILE", cached_parquet), \
             patch("data_sources.huggingface_data.CATALOG_FILE", cached_catalog):
            df, _ = download_from_hf(
                ["NY.GDP.MKTP.CD"],
                ["USA", "GBR"],
                2020, 2020,
            )

        assert set(df["year"].unique()) == {2020}

    def test_reports_missing_indicators(self, cached_parquet, cached_catalog):
        with patch("data_sources.huggingface_data.CACHE_FILE", cached_parquet), \
             patch("data_sources.huggingface_data.CATALOG_FILE", cached_catalog):
            df, failed = download_from_hf(
                ["NY.GDP.MKTP.CD", "FAKE.INDICATOR"],
                ["USA"],
                2019, 2020,
            )

        assert len(failed) == 1
        assert "NY.GDP.MKTP.CD" in df.columns

    def test_raises_when_no_cache(self, tmp_path):
        with patch("data_sources.huggingface_data.CACHE_FILE", tmp_path / "missing.parquet"):
            with pytest.raises(RuntimeError, match="cache not found"):
                download_from_hf(["NY.GDP.MKTP.CD"], ["USA"], 2020, 2020)

    def test_progress_callback_called(self, cached_parquet, cached_catalog):
        callback = MagicMock()
        with patch("data_sources.huggingface_data.CACHE_FILE", cached_parquet), \
             patch("data_sources.huggingface_data.CATALOG_FILE", cached_catalog):
            download_from_hf(
                ["NY.GDP.MKTP.CD"],
                ["USA"],
                2019, 2020,
                progress_callback=callback,
            )
        assert callback.call_count >= 1

    def test_all_missing_raises(self, cached_parquet, cached_catalog):
        with patch("data_sources.huggingface_data.CACHE_FILE", cached_parquet), \
             patch("data_sources.huggingface_data.CATALOG_FILE", cached_catalog):
            with pytest.raises(RuntimeError, match="All indicators failed"):
                download_from_hf(
                    ["FAKE.ONE", "FAKE.TWO"],
                    ["USA"],
                    2019, 2020,
                )


# ---------------------------------------------------------------------------
# download_hf_dataset
# ---------------------------------------------------------------------------

class TestDownloadHfDataset:
    def test_skips_when_cached(self, cached_parquet):
        with patch("data_sources.huggingface_data.CACHE_FILE", cached_parquet), \
             patch("data_sources.huggingface_data.CACHE_DIR", cached_parquet.parent):
            result = download_hf_dataset(force=False)
            assert result == cached_parquet

    @patch("data_sources.huggingface_data.pd.read_parquet")
    def test_downloads_when_forced(self, mock_read, cached_parquet, sample_long_df, cache_dir):
        mock_read.return_value = sample_long_df
        catalog_path = cache_dir / "indicator_catalog.json"

        with patch("data_sources.huggingface_data.CACHE_FILE", cached_parquet), \
             patch("data_sources.huggingface_data.CACHE_DIR", cache_dir), \
             patch("data_sources.huggingface_data.CATALOG_FILE", catalog_path):
            result = download_hf_dataset(force=True)

        mock_read.assert_called_once()
        assert result == cached_parquet
        assert catalog_path.exists()

    @patch("data_sources.huggingface_data.pd.read_parquet")
    def test_downloads_when_no_cache(self, mock_read, tmp_path, sample_long_df):
        cache_dir = tmp_path / "hf_cache"
        cache_file = cache_dir / "wdi.parquet"
        catalog_file = cache_dir / "indicator_catalog.json"
        mock_read.return_value = sample_long_df

        with patch("data_sources.huggingface_data.CACHE_FILE", cache_file), \
             patch("data_sources.huggingface_data.CACHE_DIR", cache_dir), \
             patch("data_sources.huggingface_data.CATALOG_FILE", catalog_file):
            result = download_hf_dataset()

        mock_read.assert_called_once()
        assert cache_file.exists()
        assert catalog_file.exists()
