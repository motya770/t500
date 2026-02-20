"""Tests for ui/page_download.py module."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, PropertyMock


class TestPageDownloadRender:
    """Tests for the download page render() function."""

    @patch("ui.page_download.save_dataset")
    @patch("ui.page_download.download_multiple_indicators")
    @patch("ui.page_download.get_country_groups")
    @patch("ui.page_download.get_all_indicators")
    @patch("ui.page_download.st")
    def test_render_no_indicators_selected(self, mock_st, mock_all_ind, mock_groups, mock_download, mock_save):
        """When no indicators are selected, a warning is shown."""
        mock_all_ind.return_value = {"NY.GDP.MKTP.CD": "GDP"}
        mock_groups.return_value = {"G7": ["USA", "GBR"]}
        mock_st.session_state = {}

        # Mock UI components to return no selections
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.checkbox.return_value = False
        mock_st.radio.return_value = "Predefined groups"
        mock_st.multiselect.return_value = ["G7"]
        mock_st.number_input.side_effect = [2000, 2023]
        mock_st.text_input.return_value = "economic_data"

        from ui.page_download import render
        render()

        # Should warn about no indicators
        mock_st.warning.assert_called()

    @patch("ui.page_download.save_dataset")
    @patch("ui.page_download.download_multiple_indicators")
    @patch("ui.page_download.get_country_groups")
    @patch("ui.page_download.get_all_indicators")
    @patch("ui.page_download.INDICATOR_CATEGORIES", {"GDP": {"NY.GDP.MKTP.CD": "GDP"}})
    @patch("ui.page_download.st")
    def test_render_start_year_after_end_year(self, mock_st, mock_all_ind, mock_groups, mock_download, mock_save):
        """When start year > end year, an error is shown."""
        mock_all_ind.return_value = {"NY.GDP.MKTP.CD": "GDP"}
        mock_groups.return_value = {"G7": ["USA"]}
        mock_st.session_state = {}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.checkbox.return_value = False
        mock_st.radio.return_value = "Predefined groups"
        mock_st.multiselect.return_value = ["G7"]
        # Start year > end year
        mock_st.number_input.side_effect = [2023, 2000]
        mock_st.text_input.return_value = "test"
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

        from ui.page_download import render
        render()

        mock_st.error.assert_called_with("Start year must be before end year.")

    @patch("ui.page_download.save_dataset")
    @patch("ui.page_download.download_multiple_indicators")
    @patch("ui.page_download.get_country_groups")
    @patch("ui.page_download.get_all_indicators")
    @patch("ui.page_download.INDICATOR_CATEGORIES", {"GDP": {"NY.GDP.MKTP.CD": "GDP"}})
    @patch("ui.page_download.st")
    def test_render_successful_download(self, mock_st, mock_all_ind, mock_groups, mock_download, mock_save):
        """Successful download stores data in session state."""
        mock_all_ind.return_value = {"NY.GDP.MKTP.CD": "GDP"}
        mock_groups.return_value = {"G7": ["USA"]}

        session_state = {}
        mock_st.session_state = session_state
        mock_st.columns.return_value = [MagicMock(), MagicMock()]

        # Simulate that the checkbox for the indicator is checked
        mock_st.checkbox.return_value = True
        mock_st.radio.return_value = "Predefined groups"
        mock_st.multiselect.return_value = ["G7"]
        mock_st.number_input.side_effect = [2020, 2021]
        mock_st.text_input.return_value = "test_data"
        mock_st.button.return_value = True

        # Mock progress bar and status
        mock_progress = MagicMock()
        mock_st.progress.return_value = mock_progress
        mock_st.empty.return_value = MagicMock()
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

        # Mock downloaded data
        df = pd.DataFrame({
            "country": ["USA", "USA"],
            "year": [2020, 2021],
            "NY.GDP.MKTP.CD": [21000.0, 22000.0],
        })
        mock_download.return_value = df
        from pathlib import Path
        mock_save.return_value = Path("data/test_data.csv")

        from ui.page_download import render
        render()

        # Should store data in session state
        assert "current_dataset" in session_state
        assert "indicator_names" in session_state

    @patch("ui.page_download.save_dataset")
    @patch("ui.page_download.download_multiple_indicators")
    @patch("ui.page_download.get_country_groups")
    @patch("ui.page_download.get_all_indicators")
    @patch("ui.page_download.INDICATOR_CATEGORIES", {"GDP": {"NY.GDP.MKTP.CD": "GDP"}})
    @patch("ui.page_download.st")
    def test_render_download_returns_empty(self, mock_st, mock_all_ind, mock_groups, mock_download, mock_save):
        """If download returns empty DataFrame, an error is shown."""
        mock_all_ind.return_value = {"NY.GDP.MKTP.CD": "GDP"}
        mock_groups.return_value = {"G7": ["USA"]}
        mock_st.session_state = {}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.checkbox.return_value = True
        mock_st.radio.return_value = "Predefined groups"
        mock_st.multiselect.return_value = ["G7"]
        mock_st.number_input.side_effect = [2020, 2021]
        mock_st.text_input.return_value = "test"
        mock_st.button.return_value = True
        mock_st.progress.return_value = MagicMock()
        mock_st.empty.return_value = MagicMock()
        mock_download.return_value = pd.DataFrame()

        from ui.page_download import render
        render()

        mock_st.error.assert_called_with("No data returned. Try different indicators or countries.")

    @patch("ui.page_download.save_dataset")
    @patch("ui.page_download.download_multiple_indicators")
    @patch("ui.page_download.get_country_groups")
    @patch("ui.page_download.get_all_indicators")
    @patch("ui.page_download.INDICATOR_CATEGORIES", {"GDP": {"NY.GDP.MKTP.CD": "GDP"}})
    @patch("ui.page_download.st")
    def test_render_download_exception(self, mock_st, mock_all_ind, mock_groups, mock_download, mock_save):
        """If download raises an exception, error is shown."""
        mock_all_ind.return_value = {"NY.GDP.MKTP.CD": "GDP"}
        mock_groups.return_value = {"G7": ["USA"]}
        mock_st.session_state = {}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.checkbox.return_value = True
        mock_st.radio.return_value = "Predefined groups"
        mock_st.multiselect.return_value = ["G7"]
        mock_st.number_input.side_effect = [2020, 2021]
        mock_st.text_input.return_value = "test"
        mock_st.button.return_value = True
        mock_st.progress.return_value = MagicMock()
        mock_st.empty.return_value = MagicMock()
        mock_download.side_effect = RuntimeError("API error")

        from ui.page_download import render
        render()

        mock_st.error.assert_called()

    @patch("ui.page_download.get_country_groups")
    @patch("ui.page_download.get_all_indicators")
    @patch("ui.page_download.INDICATOR_CATEGORIES", {"GDP": {"NY.GDP.MKTP.CD": "GDP"}})
    @patch("ui.page_download.st")
    def test_render_manual_country_entry(self, mock_st, mock_all_ind, mock_groups):
        """Manual country entry mode parses comma-separated codes."""
        mock_all_ind.return_value = {"NY.GDP.MKTP.CD": "GDP"}
        mock_groups.return_value = {}
        mock_st.session_state = {}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.checkbox.return_value = False
        mock_st.radio.return_value = "Manual entry"
        mock_st.text_area.return_value = "USA, GBR, DEU"
        mock_st.number_input.side_effect = [2020, 2023]
        mock_st.text_input.return_value = "test"

        from ui.page_download import render
        render()

        # Should call text_area for manual input
        mock_st.text_area.assert_called_once()
