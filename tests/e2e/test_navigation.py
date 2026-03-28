"""Tests for sidebar navigation and page routing."""

import pytest
from tests.e2e.conftest import click_sidebar_nav


SIDEBAR_PAGES = [
    "Download Data",
    "Stock / ETF Data",
    "Explore & Visualize",
    "Correlation Analysis",
    "Inflation-Stock Models",
    "Cargo Plane Analysis",
    "Oil Tanker Analysis",
    "News Sentiment",
]


class TestSidebarNavigation:
    """Verify the sidebar renders with all navigation items and routing works."""

    def test_sidebar_present(self, app_page):
        """Sidebar section should exist in the DOM."""
        sidebar = app_page.locator("section[data-testid='stSidebar']")
        assert sidebar.count() >= 1

    def test_sidebar_radio_present(self, app_page):
        """Sidebar should contain a radio widget for navigation."""
        radio = app_page.locator("[data-testid='stRadio']")
        assert radio.count() >= 1

    def test_sidebar_contains_all_pages(self, app_page):
        """Every navigation entry should appear in the sidebar radio."""
        radio = app_page.locator("[data-testid='stRadio']")
        for label in SIDEBAR_PAGES:
            assert radio.get_by_text(label).count() >= 1, (
                f"Sidebar should contain '{label}'"
            )

    def test_default_page_is_download(self, app_page):
        """The first page (Download Data) should load by default."""
        header = app_page.get_by_text("Download Economic Data")
        assert header.count() >= 1

    @pytest.mark.parametrize("page_label,expected_header", [
        ("Download Data", "Download Economic Data"),
        ("Explore & Visualize", "Explore & Visualize Data"),
        ("Correlation Analysis", "Correlation & ML Analysis"),
    ])
    def test_navigate_to_page(self, app_page, page_label, expected_header):
        """Clicking a sidebar nav item should load the corresponding page."""
        click_sidebar_nav(app_page, page_label)
        # Wait for the expected header to appear (some pages import heavy
        # ML libraries on first load, so allow up to 30 seconds).
        app_page.wait_for_selector(f"text={expected_header}", timeout=30000)
        assert app_page.get_by_text(expected_header).count() >= 1, (
            f"Expected header '{expected_header}' after navigating to '{page_label}'"
        )

    def test_route_select_label_visible(self, app_page):
        """The sidebar should show the 'Route Select' label."""
        radio = app_page.locator("[data-testid='stRadio']")
        assert radio.get_by_text("Route Select").count() >= 1
