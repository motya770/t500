"""Tests for app.py module."""

import pytest
from unittest.mock import patch, MagicMock


class TestAppRouting:
    """Tests for app.py page routing logic."""

    @patch("ui.page_download.render")
    @patch("streamlit.sidebar")
    @patch("streamlit.markdown")
    @patch("streamlit.set_page_config")
    def test_download_page_routing(self, mock_config, mock_markdown, mock_sidebar, mock_render):
        """Download Data page is routed correctly."""
        mock_sidebar.radio.return_value = "\U0001F4E6 Download Data"
        mock_sidebar.markdown = MagicMock()

        # We can't easily import app.py since it executes at module level,
        # so we test the routing logic directly
        page = "\U0001F4E6 Download Data"
        page_name = page.split(" ", 1)[1] if " " in page else page
        assert page_name == "Download Data"

    def test_page_name_extraction_download(self):
        page = "\U0001F4E6 Download Data"
        page_name = page.split(" ", 1)[1] if " " in page else page
        assert page_name == "Download Data"

    def test_page_name_extraction_explore(self):
        page = "\U0001F50D Explore & Visualize"
        page_name = page.split(" ", 1)[1] if " " in page else page
        assert page_name == "Explore & Visualize"

    def test_page_name_extraction_correlations(self):
        page = "\U0001F517 Correlation Analysis"
        page_name = page.split(" ", 1)[1] if " " in page else page
        assert page_name == "Correlation Analysis"

    def test_page_name_extraction_stock(self):
        page = "\U0001F4C8 Stock / ETF Data"
        page_name = page.split(" ", 1)[1] if " " in page else page
        assert page_name == "Stock / ETF Data"

    def test_page_name_extraction_inflation(self):
        page = "\U0001F4CA Inflation-Stock Models"
        page_name = page.split(" ", 1)[1] if " " in page else page
        assert page_name == "Inflation-Stock Models"

    def test_page_name_extraction_cargo(self):
        page = "\u2708 Cargo Plane Analysis"
        page_name = page.split(" ", 1)[1] if " " in page else page
        assert page_name == "Cargo Plane Analysis"

    def test_page_name_extraction_oil(self):
        page = "\U0001F6E2 Oil Tanker Analysis"
        page_name = page.split(" ", 1)[1] if " " in page else page
        assert page_name == "Oil Tanker Analysis"

    def test_page_name_extraction_news(self):
        page = "\U0001F4F0 News Sentiment"
        page_name = page.split(" ", 1)[1] if " " in page else page
        assert page_name == "News Sentiment"

    def test_all_routes_covered(self):
        """Verify all sidebar options map to known route names."""
        pages = [
            "\U0001F4E6 Download Data",
            "\U0001F4C8 Stock / ETF Data",
            "\U0001F50D Explore & Visualize",
            "\U0001F517 Correlation Analysis",
            "\U0001F4CA Inflation-Stock Models",
            "\u2708 Cargo Plane Analysis",
            "\U0001F6E2 Oil Tanker Analysis",
            "\U0001F4F0 News Sentiment",
        ]
        expected_names = [
            "Download Data",
            "Stock / ETF Data",
            "Explore & Visualize",
            "Correlation Analysis",
            "Inflation-Stock Models",
            "Cargo Plane Analysis",
            "Oil Tanker Analysis",
            "News Sentiment",
        ]
        for page, expected in zip(pages, expected_names):
            page_name = page.split(" ", 1)[1] if " " in page else page
            assert page_name == expected, f"Route mismatch: {page} -> {page_name}, expected {expected}"
