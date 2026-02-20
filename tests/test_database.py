"""Tests for data_sources/database.py module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch

from data_sources.database import (
    init_db,
    save_economic_dataset,
    load_economic_dataset,
    save_stock_db,
    load_stock_db,
    save_generic_dataset,
    load_generic_dataset,
    save_dataset_smart,
    load_dataset_smart,
    list_datasets,
    delete_dataset,
    save_news_articles,
    load_news_articles,
    save_indicator_metadata,
    migrate_csv_to_db,
    DB_PATH,
)


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Redirect DB_PATH to a temp directory for each test."""
    db_path = tmp_path / "test.db"
    with patch("data_sources.database.DB_PATH", db_path), \
         patch("data_sources.database.DB_DIR", tmp_path):
        init_db()
        yield tmp_path, db_path


class TestInitDb:
    def test_creates_database_file(self, temp_db):
        tmp_path, db_path = temp_db
        assert db_path.exists()

    def test_idempotent(self, temp_db):
        # Calling init_db again should not raise
        init_db()


class TestEconomicData:
    def test_save_and_load_round_trip(self, temp_db):
        df = pd.DataFrame({
            "country": ["USA", "USA", "GBR", "GBR"],
            "year": [2020, 2021, 2020, 2021],
            "NY.GDP.MKTP.CD": [21000.0, 22000.0, 3000.0, 3100.0],
            "FP.CPI.TOTL.ZG": [1.2, 5.4, 0.9, 2.1],
        })

        save_economic_dataset(df, "test_econ")
        loaded = load_economic_dataset("test_econ")

        assert len(loaded) == 4
        assert "country" in loaded.columns
        assert "year" in loaded.columns
        assert "NY.GDP.MKTP.CD" in loaded.columns
        assert "FP.CPI.TOTL.ZG" in loaded.columns

    def test_values_preserved(self, temp_db):
        df = pd.DataFrame({
            "country": ["USA"],
            "year": [2020],
            "NY.GDP.MKTP.CD": [21000.5],
        })

        save_economic_dataset(df, "test_val")
        loaded = load_economic_dataset("test_val")

        assert loaded["NY.GDP.MKTP.CD"].iloc[0] == pytest.approx(21000.5)

    def test_sparse_data_handled(self, temp_db):
        df = pd.DataFrame({
            "country": ["USA", "GBR"],
            "year": [2020, 2020],
            "IND1": [100.0, np.nan],
            "IND2": [np.nan, 200.0],
        })

        save_economic_dataset(df, "sparse")
        loaded = load_economic_dataset("sparse")

        assert len(loaded) == 2
        # USA should have IND1 but NaN for IND2
        usa_row = loaded[loaded["country"] == "USA"].iloc[0]
        assert usa_row["IND1"] == pytest.approx(100.0)
        assert pd.isna(usa_row["IND2"])

    def test_load_nonexistent_raises(self, temp_db):
        with pytest.raises(FileNotFoundError):
            load_economic_dataset("nonexistent")

    def test_overwrite_existing(self, temp_db):
        df1 = pd.DataFrame({
            "country": ["USA"], "year": [2020], "IND1": [100.0],
        })
        df2 = pd.DataFrame({
            "country": ["USA"], "year": [2020], "IND1": [999.0],
        })

        save_economic_dataset(df1, "overwrite_test")
        save_economic_dataset(df2, "overwrite_test")
        loaded = load_economic_dataset("overwrite_test")

        assert loaded["IND1"].iloc[0] == pytest.approx(999.0)


class TestStockData:
    def test_save_and_load_round_trip(self, temp_db):
        df = pd.DataFrame({
            "ticker": ["VOO", "VOO"],
            "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "year": [2020, 2020],
            "month": [1, 2],
            "open": [300.0, 305.0],
            "high": [310.0, 315.0],
            "low": [295.0, 300.0],
            "close": [308.0, 312.0],
            "volume": [1000000, 1100000],
        })

        save_stock_db(df, "test_stock")
        loaded = load_stock_db("test_stock")

        assert len(loaded) == 2
        assert "ticker" in loaded.columns
        assert "close" in loaded.columns
        assert loaded["close"].iloc[0] == pytest.approx(308.0)

    def test_date_is_datetime(self, temp_db):
        df = pd.DataFrame({
            "ticker": ["VOO"],
            "date": pd.to_datetime(["2020-01-01"]),
            "year": [2020],
            "month": [1],
            "open": [300.0],
            "high": [310.0],
            "low": [295.0],
            "close": [308.0],
            "volume": [1000000],
        })

        save_stock_db(df, "dt_test")
        loaded = load_stock_db("dt_test")

        assert pd.api.types.is_datetime64_any_dtype(loaded["date"])

    def test_load_nonexistent_raises(self, temp_db):
        with pytest.raises(FileNotFoundError):
            load_stock_db("nonexistent")

    def test_multiple_tickers(self, temp_db):
        df = pd.DataFrame({
            "ticker": ["VOO", "QQQ"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "year": [2020, 2020],
            "month": [1, 1],
            "open": [300.0, 250.0],
            "high": [310.0, 260.0],
            "low": [295.0, 245.0],
            "close": [308.0, 255.0],
            "volume": [1000000, 900000],
        })

        save_stock_db(df, "multi_ticker")
        loaded = load_stock_db("multi_ticker")

        assert set(loaded["ticker"].unique()) == {"VOO", "QQQ"}


class TestGenericData:
    def test_save_and_load_round_trip(self, temp_db):
        df = pd.DataFrame({
            "country": ["USA", "USA"],
            "year": [2020, 2021],
            "VOO_annual_return_pct": [15.5, -3.2],
            "GDP": [21000.0, 22000.0],
        })

        save_generic_dataset(df, "test_merged")
        loaded = load_generic_dataset("test_merged")

        assert len(loaded) == 2
        assert "VOO_annual_return_pct" in loaded.columns

    def test_column_order_preserved(self, temp_db):
        df = pd.DataFrame({
            "z_col": [1],
            "a_col": [2],
            "m_col": [3],
        })

        save_generic_dataset(df, "col_order")
        loaded = load_generic_dataset("col_order")

        assert list(loaded.columns) == ["z_col", "a_col", "m_col"]

    def test_load_nonexistent_raises(self, temp_db):
        with pytest.raises(FileNotFoundError):
            load_generic_dataset("nonexistent")


class TestSmartSaveLoad:
    def test_economic_data_detected(self, temp_db):
        df = pd.DataFrame({
            "country": ["USA"], "year": [2020], "IND1": [100.0],
        })

        save_dataset_smart(df, "smart_econ")
        assert "smart_econ" in list_datasets()

        loaded = load_dataset_smart("smart_econ")
        assert "country" in loaded.columns

    def test_stock_data_detected(self, temp_db):
        df = pd.DataFrame({
            "ticker": ["VOO"],
            "date": pd.to_datetime(["2020-01-01"]),
            "year": [2020],
            "month": [1],
            "close": [300.0],
        })

        save_dataset_smart(df, "smart_stock")
        loaded = load_dataset_smart("smart_stock")

        assert "ticker" in loaded.columns

    def test_generic_fallback(self, temp_db):
        df = pd.DataFrame({
            "arbitrary_col": [1, 2, 3],
            "another": ["a", "b", "c"],
        })

        save_dataset_smart(df, "smart_generic")
        loaded = load_dataset_smart("smart_generic")

        assert len(loaded) == 3
        assert "arbitrary_col" in loaded.columns

    def test_load_nonexistent_raises(self, temp_db):
        with pytest.raises(FileNotFoundError):
            load_dataset_smart("nonexistent")


class TestListAndDelete:
    def test_list_empty(self, temp_db):
        assert list_datasets() == []

    def test_list_after_save(self, temp_db):
        df = pd.DataFrame({"country": ["USA"], "year": [2020], "IND1": [1.0]})
        save_economic_dataset(df, "ds1")
        save_economic_dataset(df, "ds2")

        names = list_datasets()
        assert "ds1" in names
        assert "ds2" in names

    def test_list_filtered_by_type(self, temp_db):
        econ_df = pd.DataFrame({
            "country": ["USA"], "year": [2020], "IND1": [1.0],
        })
        stock_df = pd.DataFrame({
            "ticker": ["VOO"],
            "date": pd.to_datetime(["2020-01-01"]),
            "year": [2020], "month": [1], "close": [300.0],
        })

        save_dataset_smart(econ_df, "econ_ds")
        save_dataset_smart(stock_df, "stock_ds")

        econ_list = list_datasets(data_type="economic")
        stock_list = list_datasets(data_type="stock")

        assert "econ_ds" in econ_list
        assert "stock_ds" not in econ_list
        assert "stock_ds" in stock_list
        assert "econ_ds" not in stock_list

    def test_delete_dataset(self, temp_db):
        df = pd.DataFrame({"country": ["USA"], "year": [2020], "IND1": [1.0]})
        save_economic_dataset(df, "to_delete")

        assert "to_delete" in list_datasets()
        delete_dataset("to_delete")
        assert "to_delete" not in list_datasets()

    def test_delete_cascades_data(self, temp_db):
        df = pd.DataFrame({"country": ["USA"], "year": [2020], "IND1": [1.0]})
        save_economic_dataset(df, "cascade_test")
        delete_dataset("cascade_test")

        with pytest.raises(FileNotFoundError):
            load_economic_dataset("cascade_test")


class TestNewsArticles:
    def test_save_and_load(self, temp_db):
        articles = [
            {
                "title": "Test Article",
                "summary": "A test summary",
                "url": "https://example.com",
                "source": "Test Source",
                "published": datetime(2024, 1, 15, 10, 30),
            },
        ]

        save_news_articles(articles)
        loaded = load_news_articles()

        assert len(loaded) == 1
        assert loaded["title"].iloc[0] == "Test Article"

    def test_save_multiple_articles(self, temp_db):
        articles = [
            {
                "title": f"Article {i}",
                "summary": f"Summary {i}",
                "url": f"https://example.com/{i}",
                "source": "Test",
                "published": datetime(2024, 1, i + 1),
            }
            for i in range(5)
        ]

        save_news_articles(articles)
        loaded = load_news_articles()

        assert len(loaded) == 5

    def test_load_with_limit(self, temp_db):
        articles = [
            {
                "title": f"Article {i}",
                "summary": "",
                "url": "",
                "source": "Test",
                "published": datetime(2024, 1, i + 1),
            }
            for i in range(10)
        ]

        save_news_articles(articles)
        loaded = load_news_articles(limit=3)

        assert len(loaded) == 3


class TestIndicatorMetadata:
    def test_save_metadata(self, temp_db):
        categories = {
            "GDP & Growth": {
                "NY.GDP.MKTP.CD": "GDP (current US$)",
                "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
            },
        }

        save_indicator_metadata(categories)
        # Verify by saving and loading -- metadata is stored

    def test_upsert_metadata(self, temp_db):
        categories = {
            "Cat1": {"CODE1": "Name 1"},
        }
        save_indicator_metadata(categories)

        updated = {
            "Cat1": {"CODE1": "Updated Name"},
        }
        save_indicator_metadata(updated)
        # Should not raise (ON CONFLICT DO UPDATE)


class TestMigration:
    def test_migrate_csv_files(self, temp_db):
        tmp_path, _ = temp_db

        # Create a CSV file in the data dir
        csv_path = tmp_path / "legacy_data.csv"
        df = pd.DataFrame({
            "country": ["USA", "GBR"],
            "year": [2020, 2020],
            "NY.GDP.MKTP.CD": [21000.0, 3000.0],
        })
        df.to_csv(csv_path, index=False)

        with patch("data_sources.database.Path") as mock_path_cls:
            # Make the migration function find our temp dir
            mock_path_cls.return_value.parent.parent.__truediv__ = lambda s, x: tmp_path
            # Actually just patch the csv_dir computation
            migrated = migrate_csv_to_db()

        # The dataset should now be in the database
        assert "legacy_data" in list_datasets()

    def test_migrate_skips_existing(self, temp_db):
        tmp_path, _ = temp_db

        # Save to DB first
        df = pd.DataFrame({"country": ["USA"], "year": [2020], "IND1": [1.0]})
        save_economic_dataset(df, "already_there")

        # Create CSV with same name
        csv_path = tmp_path / "already_there.csv"
        df.to_csv(csv_path, index=False)

        migrated = migrate_csv_to_db()
        # Should have skipped the existing one
        assert migrated == 0

    def test_migrate_empty_dir(self, temp_db):
        migrated = migrate_csv_to_db()
        assert migrated == 0
