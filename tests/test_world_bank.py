"""Tests for data_sources/world_bank.py module."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

from data_sources.world_bank import (
    INDICATOR_CATEGORIES,
    get_all_indicators,
    get_country_groups,
    download_indicator,
    download_multiple_indicators,
    save_dataset,
    load_dataset,
    list_saved_datasets,
    DATA_DIR,
)


class TestIndicatorCategories:
    """Tests for indicator category constants."""

    def test_categories_not_empty(self):
        assert len(INDICATOR_CATEGORIES) > 0

    def test_all_categories_have_indicators(self):
        for cat, indicators in INDICATOR_CATEGORIES.items():
            assert len(indicators) > 0, f"Category '{cat}' has no indicators"

    def test_indicator_codes_are_strings(self):
        for cat, indicators in INDICATOR_CATEGORIES.items():
            for code, desc in indicators.items():
                assert isinstance(code, str), f"Code {code} in '{cat}' is not a string"
                assert isinstance(desc, str), f"Description for {code} is not a string"
                assert len(code) > 0
                assert len(desc) > 0

    def test_known_categories_exist(self):
        expected = [
            "GDP & Growth",
            "Trade",
            "Inflation & Prices",
            "Employment",
            "Financial",
        ]
        for cat in expected:
            assert cat in INDICATOR_CATEGORIES, f"Expected category '{cat}' not found"

    def test_gdp_indicator_exists(self):
        assert "NY.GDP.MKTP.CD" in INDICATOR_CATEGORIES["GDP & Growth"]


class TestGetAllIndicators:
    """Tests for get_all_indicators()."""

    def test_returns_dict(self):
        result = get_all_indicators()
        assert isinstance(result, dict)

    def test_contains_all_unique_indicators(self):
        result = get_all_indicators()
        # Some indicators appear in multiple categories, so count unique codes
        all_codes = set()
        for indicators in INDICATOR_CATEGORIES.values():
            all_codes.update(indicators.keys())
        assert len(result) == len(all_codes)

    def test_values_are_descriptions(self):
        result = get_all_indicators()
        for code, desc in result.items():
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_all_category_indicators_in_flat_dict(self):
        result = get_all_indicators()
        for cat, indicators in INDICATOR_CATEGORIES.items():
            for code in indicators:
                assert code in result, f"Code {code} from '{cat}' missing in get_all_indicators()"


class TestGetCountryGroups:
    """Tests for get_country_groups()."""

    def test_returns_dict(self):
        result = get_country_groups()
        assert isinstance(result, dict)

    def test_known_groups_exist(self):
        result = get_country_groups()
        assert "G7" in result
        assert "BRICS" in result

    def test_g7_countries(self):
        result = get_country_groups()
        g7 = result["G7"]
        assert "USA" in g7
        assert "GBR" in g7
        assert "JPN" in g7
        assert len(g7) == 7

    def test_all_groups_have_countries(self):
        result = get_country_groups()
        for group, countries in result.items():
            assert len(countries) > 0, f"Group '{group}' has no countries"

    def test_country_codes_are_3_letter(self):
        result = get_country_groups()
        for group, countries in result.items():
            for code in countries:
                assert len(code) == 3, f"Country code '{code}' in '{group}' is not 3 letters"


class TestDownloadIndicator:
    """Tests for download_indicator() with mocked API calls."""

    @patch("data_sources.world_bank.wb")
    def test_returns_dataframe(self, mock_wb):
        mock_data = pd.DataFrame(
            {"YR2020": [100.0, 200.0], "YR2021": [110.0, 210.0]},
            index=["USA", "GBR"],
        )
        mock_data.index.name = "economy"
        mock_wb.data.DataFrame.return_value = mock_data

        result = download_indicator("NY.GDP.MKTP.CD", ["USA", "GBR"], 2020, 2021)

        assert isinstance(result, pd.DataFrame)
        assert "country" in result.columns
        assert "year" in result.columns
        assert "value" in result.columns

    @patch("data_sources.world_bank.wb")
    def test_empty_response(self, mock_wb):
        mock_wb.data.DataFrame.return_value = pd.DataFrame()

        result = download_indicator("NY.GDP.MKTP.CD", ["USA"], 2020, 2021)
        assert result.empty
        assert list(result.columns) == ["country", "year", "value"]

    @patch("data_sources.world_bank.wb")
    def test_api_error_raises_runtime(self, mock_wb):
        mock_wb.data.DataFrame.side_effect = Exception("API error")

        with pytest.raises(RuntimeError, match="Failed to download indicator"):
            download_indicator("INVALID", ["USA"], 2020, 2021)

    @patch("data_sources.world_bank.wb")
    def test_drops_nan_values(self, mock_wb):
        import numpy as np
        mock_data = pd.DataFrame(
            {"YR2020": [100.0, np.nan], "YR2021": [np.nan, 210.0]},
            index=["USA", "GBR"],
        )
        mock_data.index.name = "economy"
        mock_wb.data.DataFrame.return_value = mock_data

        result = download_indicator("NY.GDP.MKTP.CD", ["USA", "GBR"], 2020, 2021)
        assert not result["value"].isna().any()

    @patch("data_sources.world_bank.wb")
    def test_year_parsing(self, mock_wb):
        mock_data = pd.DataFrame(
            {"YR2019": [50.0], "YR2020": [100.0]},
            index=["USA"],
        )
        mock_data.index.name = "economy"
        mock_wb.data.DataFrame.return_value = mock_data

        result = download_indicator("NY.GDP.MKTP.CD", ["USA"], 2019, 2020)
        assert set(result["year"].unique()) == {2019, 2020}


class TestDownloadMultipleIndicators:
    """Tests for download_multiple_indicators() with mocked API calls."""

    @patch("data_sources.world_bank.download_indicator")
    def test_merges_multiple_indicators(self, mock_download):
        df1 = pd.DataFrame({
            "country": ["USA", "USA"],
            "year": [2020, 2021],
            "value": [100.0, 110.0],
        })
        df2 = pd.DataFrame({
            "country": ["USA", "USA"],
            "year": [2020, 2021],
            "value": [5.0, 4.5],
        })
        mock_download.side_effect = [df1, df2]

        result, failed = download_multiple_indicators(
            ["NY.GDP.MKTP.CD", "FP.CPI.TOTL.ZG"],
            ["USA"],
            2020,
            2021,
        )

        assert isinstance(result, pd.DataFrame)
        assert failed == []
        assert "country" in result.columns
        assert "year" in result.columns
        assert "NY.GDP.MKTP.CD" in result.columns
        assert "FP.CPI.TOTL.ZG" in result.columns

    @patch("data_sources.world_bank.download_indicator")
    def test_handles_empty_indicator(self, mock_download):
        df1 = pd.DataFrame({
            "country": ["USA"],
            "year": [2020],
            "value": [100.0],
        })
        mock_download.side_effect = [
            df1,
            pd.DataFrame(columns=["country", "year", "value"]),
        ]

        result, failed = download_multiple_indicators(["IND1", "IND2"], ["USA"], 2020, 2020)
        assert "IND1" in result.columns
        assert failed == []

    @patch("data_sources.world_bank.download_indicator")
    def test_all_empty_returns_empty(self, mock_download):
        mock_download.return_value = pd.DataFrame(columns=["country", "year", "value"])

        result, failed = download_multiple_indicators(["IND1"], ["USA"], 2020, 2020)
        assert result.empty
        assert failed == []

    @patch("data_sources.world_bank.download_indicator")
    def test_progress_callback(self, mock_download):
        df = pd.DataFrame({
            "country": ["USA"],
            "year": [2020],
            "value": [100.0],
        })
        mock_download.return_value = df

        callback = MagicMock()
        download_multiple_indicators(["IND1", "IND2"], ["USA"], 2020, 2020, progress_callback=callback)

        assert callback.call_count == 2

    @patch("data_sources.world_bank.download_indicator")
    def test_skips_failed_indicators(self, mock_download):
        """Failed indicators are skipped and reported, not fatal."""
        df1 = pd.DataFrame({
            "country": ["USA"],
            "year": [2020],
            "value": [100.0],
        })
        mock_download.side_effect = [
            df1,
            RuntimeError("API error"),
        ]

        result, failed = download_multiple_indicators(
            ["NY.GDP.MKTP.CD", "GC.BAL.CASH.GD.ZS"],
            ["USA"],
            2020,
            2020,
        )

        assert "NY.GDP.MKTP.CD" in result.columns
        assert len(failed) == 1


class TestDataPersistence:
    """Tests for save_dataset, load_dataset, list_saved_datasets."""

    def test_save_and_load(self, tmp_path):
        df = pd.DataFrame({"country": ["USA"], "year": [2020], "value": [100.0]})

        with patch("data_sources.database.DB_PATH", tmp_path / "test.db"), \
             patch("data_sources.database.DB_DIR", tmp_path), \
             patch("data_sources.world_bank.DATA_DIR", tmp_path):
            from data_sources.database import init_db
            init_db()
            save_dataset(df, "test_data")
            loaded = load_dataset("test_data")

        assert len(loaded) == len(df)
        assert "country" in loaded.columns
        assert "year" in loaded.columns

    def test_load_nonexistent_raises(self, tmp_path):
        with patch("data_sources.database.DB_PATH", tmp_path / "test.db"), \
             patch("data_sources.database.DB_DIR", tmp_path), \
             patch("data_sources.world_bank.DATA_DIR", tmp_path):
            from data_sources.database import init_db
            init_db()
            with pytest.raises(FileNotFoundError):
                load_dataset("nonexistent")

    def test_list_saved_datasets(self, tmp_path):
        # Create some CSV files (legacy) to test fallback listing
        (tmp_path / "dataset1.csv").write_text("a,b\n1,2\n")
        (tmp_path / "dataset2.csv").write_text("a,b\n3,4\n")
        (tmp_path / "not_csv.txt").write_text("hello")

        with patch("data_sources.database.DB_PATH", tmp_path / "test.db"), \
             patch("data_sources.database.DB_DIR", tmp_path), \
             patch("data_sources.world_bank.DATA_DIR", tmp_path):
            from data_sources.database import init_db
            init_db()
            datasets = list_saved_datasets()

        assert "dataset1" in datasets
        assert "dataset2" in datasets
        assert "not_csv" not in datasets

    def test_list_empty_directory(self, tmp_path):
        with patch("data_sources.database.DB_PATH", tmp_path / "test.db"), \
             patch("data_sources.database.DB_DIR", tmp_path), \
             patch("data_sources.world_bank.DATA_DIR", tmp_path):
            from data_sources.database import init_db
            init_db()
            datasets = list_saved_datasets()
        assert datasets == []

    def test_save_creates_directory(self, tmp_path):
        new_dir = tmp_path / "new_subdir"
        df = pd.DataFrame({"country": ["USA"], "year": [2020], "value": [100.0]})

        with patch("data_sources.database.DB_PATH", new_dir / "test.db"), \
             patch("data_sources.database.DB_DIR", new_dir), \
             patch("data_sources.world_bank.DATA_DIR", new_dir):
            from data_sources.database import init_db
            init_db()
            save_dataset(df, "test")

        assert new_dir.exists()

    def test_csv_fallback_on_load(self, tmp_path):
        """Legacy CSV files should still be loadable."""
        df = pd.DataFrame({"country": ["USA"], "year": [2020], "value": [100.0]})
        df.to_csv(tmp_path / "legacy.csv", index=False)

        with patch("data_sources.database.DB_PATH", tmp_path / "test.db"), \
             patch("data_sources.database.DB_DIR", tmp_path), \
             patch("data_sources.world_bank.DATA_DIR", tmp_path):
            from data_sources.database import init_db
            init_db()
            loaded = load_dataset("legacy")

        assert len(loaded) == 1
        assert loaded["country"].iloc[0] == "USA"
