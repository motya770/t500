"""Tests for ui/page_explore.py module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture
def explore_df():
    """DataFrame for explore page tests."""
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
            })
    return pd.DataFrame(rows)


class TestHelperFunctions:
    """Tests for _get_indicator_label and _indicator_columns."""

    @patch("ui.page_explore.st")
    @patch("ui.page_explore.get_all_indicators")
    def test_get_indicator_label_from_session_state(self, mock_all_ind, mock_st):
        mock_st.session_state = {"indicator_names": {"NY.GDP.MKTP.CD": "GDP"}}
        mock_all_ind.return_value = {}

        from ui.page_explore import _get_indicator_label
        result = _get_indicator_label("NY.GDP.MKTP.CD")
        assert result == "GDP"

    @patch("ui.page_explore.st")
    @patch("ui.page_explore.get_all_indicators")
    def test_get_indicator_label_from_all_indicators(self, mock_all_ind, mock_st):
        mock_st.session_state = {"indicator_names": {}}
        mock_all_ind.return_value = {"FP.CPI.TOTL.ZG": "Inflation"}

        from ui.page_explore import _get_indicator_label
        result = _get_indicator_label("FP.CPI.TOTL.ZG")
        assert result == "Inflation"

    @patch("ui.page_explore.st")
    @patch("ui.page_explore.get_all_indicators")
    def test_get_indicator_label_fallback_to_code(self, mock_all_ind, mock_st):
        mock_st.session_state = {"indicator_names": {}}
        mock_all_ind.return_value = {}

        from ui.page_explore import _get_indicator_label
        result = _get_indicator_label("UNKNOWN.CODE")
        assert result == "UNKNOWN.CODE"

    def test_indicator_columns(self, explore_df):
        from ui.page_explore import _indicator_columns
        cols = _indicator_columns(explore_df)
        assert "country" not in cols
        assert "year" not in cols
        assert "NY.GDP.MKTP.CD" in cols
        assert "FP.CPI.TOTL.ZG" in cols

    def test_indicator_columns_empty_df(self):
        from ui.page_explore import _indicator_columns
        df = pd.DataFrame({"country": [], "year": []})
        cols = _indicator_columns(df)
        assert cols == []


class TestRenderPage:
    """Tests for the main render() function."""

    @patch("ui.page_explore.list_saved_datasets")
    @patch("ui.page_explore.st")
    def test_render_no_data(self, mock_st, mock_list):
        """Shows info message when no datasets available."""
        mock_list.return_value = []
        mock_st.session_state = {}

        from ui.page_explore import render
        render()

        mock_st.info.assert_called()

    @patch("ui.page_explore.list_saved_datasets")
    @patch("ui.page_explore.st")
    def test_render_empty_dataset(self, mock_st, mock_list):
        """Shows warning for empty dataset."""
        mock_list.return_value = []
        mock_st.session_state = {"current_dataset": pd.DataFrame()}
        mock_st.radio.return_value = "Current session"

        from ui.page_explore import render
        render()

        mock_st.warning.assert_called()

    @patch("ui.page_explore.px")
    @patch("ui.page_explore.apply_steam_style")
    @patch("ui.page_explore.list_saved_datasets")
    @patch("ui.page_explore.st")
    def test_render_with_current_session_data(self, mock_st, mock_list, mock_style, mock_px, explore_df):
        """Render succeeds with data in session state."""
        mock_list.return_value = []
        mock_st.session_state = {
            "current_dataset": explore_df,
            "current_dataset_name": "test",
            "indicator_names": {
                "NY.GDP.MKTP.CD": "GDP",
                "FP.CPI.TOTL.ZG": "Inflation",
            },
        }
        mock_st.radio.return_value = "Current session"
        # selectbox is called twice: chart type, then indicator within _render_time_series
        mock_st.selectbox.side_effect = ["Time Series", "NY.GDP.MKTP.CD"]
        mock_st.multiselect.return_value = ["USA"]

        # Mock columns and expander context managers
        mock_cols = [MagicMock() for _ in range(4)]
        mock_st.columns.return_value = mock_cols
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

        # Mock plotly figure
        mock_fig = MagicMock()
        mock_px.line.return_value = mock_fig

        from ui.page_explore import render
        render()

        # Should have called header
        mock_st.header.assert_called()

    @patch("ui.page_explore.load_dataset")
    @patch("ui.page_explore.list_saved_datasets")
    @patch("ui.page_explore.st")
    def test_render_load_saved_dataset(self, mock_st, mock_list, mock_load, explore_df):
        """Render loads saved dataset when selected."""
        mock_list.return_value = ["my_dataset"]
        mock_st.session_state = {}
        mock_st.radio.return_value = "Saved dataset"
        mock_st.selectbox.side_effect = ["my_dataset", "Time Series"]
        mock_load.return_value = explore_df

        mock_cols = [MagicMock() for _ in range(4)]
        mock_st.columns.return_value = mock_cols
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.multiselect.return_value = ["USA"]

        from ui.page_explore import render
        # This may raise due to mock complications, but we verify load was called
        try:
            render()
        except Exception:
            pass

        mock_load.assert_called_with("my_dataset")


class TestVisualizationFunctions:
    """Tests for individual visualization rendering functions."""

    @patch("ui.page_explore.apply_steam_style")
    @patch("ui.page_explore.px")
    @patch("ui.page_explore.st")
    def test_render_time_series(self, mock_st, mock_px, mock_style, explore_df):
        mock_st.session_state = {"indicator_names": {"NY.GDP.MKTP.CD": "GDP"}}
        mock_st.selectbox.return_value = "NY.GDP.MKTP.CD"
        mock_st.multiselect.return_value = ["USA", "GBR"]

        mock_fig = MagicMock()
        mock_px.line.return_value = mock_fig

        from ui.page_explore import _render_time_series
        indicators = ["NY.GDP.MKTP.CD", "FP.CPI.TOTL.ZG"]
        _render_time_series(explore_df, indicators, ["USA", "GBR", "DEU"])

        mock_px.line.assert_called_once()
        mock_st.plotly_chart.assert_called_once()

    @patch("ui.page_explore.apply_steam_style")
    @patch("ui.page_explore.px")
    @patch("ui.page_explore.st")
    def test_render_time_series_no_countries(self, mock_st, mock_px, mock_style, explore_df):
        """Early return when no countries selected."""
        mock_st.session_state = {"indicator_names": {}}
        mock_st.selectbox.return_value = "NY.GDP.MKTP.CD"
        mock_st.multiselect.return_value = []

        from ui.page_explore import _render_time_series
        _render_time_series(explore_df, ["NY.GDP.MKTP.CD"], ["USA"])

        mock_px.line.assert_not_called()

    @patch("ui.page_explore.apply_steam_style")
    @patch("ui.page_explore.px")
    @patch("ui.page_explore.st")
    def test_render_country_comparison(self, mock_st, mock_px, mock_style, explore_df):
        mock_st.session_state = {"indicator_names": {"NY.GDP.MKTP.CD": "GDP"}}
        mock_st.selectbox.return_value = "NY.GDP.MKTP.CD"
        mock_st.slider.return_value = 2020

        mock_fig = MagicMock()
        mock_px.bar.return_value = mock_fig

        from ui.page_explore import _render_country_comparison
        _render_country_comparison(explore_df, ["NY.GDP.MKTP.CD"], ["USA", "GBR"])

        mock_px.bar.assert_called_once()

    @patch("ui.page_explore.apply_steam_style")
    @patch("ui.page_explore.px")
    @patch("ui.page_explore.st")
    def test_render_scatter(self, mock_st, mock_px, mock_style, explore_df):
        mock_st.session_state = {"indicator_names": {}}
        col_mock = MagicMock()
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [col_mock, col_mock]
        mock_st.selectbox.side_effect = ["NY.GDP.MKTP.CD", "FP.CPI.TOTL.ZG"]
        mock_st.radio.return_value = "Country"

        mock_fig = MagicMock()
        mock_px.scatter.return_value = mock_fig

        from ui.page_explore import _render_scatter
        _render_scatter(explore_df, ["NY.GDP.MKTP.CD", "FP.CPI.TOTL.ZG"], ["USA"])

        mock_px.scatter.assert_called_once()

    @patch("ui.page_explore.apply_steam_style")
    @patch("ui.page_explore.px")
    @patch("ui.page_explore.st")
    def test_render_distribution(self, mock_st, mock_px, mock_style, explore_df):
        mock_st.session_state = {"indicator_names": {"NY.GDP.MKTP.CD": "GDP"}}
        mock_st.selectbox.return_value = "NY.GDP.MKTP.CD"

        mock_fig = MagicMock()
        mock_px.histogram.return_value = mock_fig

        mock_cols = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.columns.return_value = mock_cols

        from ui.page_explore import _render_distribution
        _render_distribution(explore_df, ["NY.GDP.MKTP.CD"])

        mock_px.histogram.assert_called_once()

    @patch("ui.page_explore.apply_steam_style")
    @patch("ui.page_explore.px")
    @patch("ui.page_explore.st")
    def test_render_heatmap(self, mock_st, mock_px, mock_style, explore_df):
        mock_st.session_state = {"indicator_names": {"NY.GDP.MKTP.CD": "GDP"}}
        mock_st.selectbox.return_value = "NY.GDP.MKTP.CD"

        mock_fig = MagicMock()
        mock_px.imshow.return_value = mock_fig

        from ui.page_explore import _render_heatmap
        _render_heatmap(explore_df, ["NY.GDP.MKTP.CD"], ["USA", "GBR"])

        mock_px.imshow.assert_called_once()

    @patch("ui.page_explore.apply_steam_style")
    @patch("ui.page_explore.px")
    @patch("ui.page_explore.st")
    def test_render_boxplot(self, mock_st, mock_px, mock_style, explore_df):
        mock_st.session_state = {"indicator_names": {"NY.GDP.MKTP.CD": "GDP"}}
        mock_st.selectbox.return_value = "NY.GDP.MKTP.CD"

        mock_fig = MagicMock()
        mock_px.box.return_value = mock_fig

        from ui.page_explore import _render_boxplot
        _render_boxplot(explore_df, ["NY.GDP.MKTP.CD"], ["USA", "GBR"])

        mock_px.box.assert_called_once()
