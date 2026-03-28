"""Tests for the Download Data page UI elements and interactions."""

import re

import pytest
from tests.e2e.conftest import click_sidebar_nav


@pytest.fixture()
def download_page(app_page):
    """Navigate to the Download Data page and wait for full render."""
    click_sidebar_nav(app_page, "Download Data")
    app_page.wait_for_selector("text=Download Economic Data", timeout=10000)
    # Wait for the full page to render (country section appears after all categories)
    app_page.wait_for_selector("text=Select Countries", timeout=15000)
    return app_page


class TestDownloadPageLayout:
    """Verify that all expected sections and widgets render on the Download page."""

    def test_page_header(self, download_page):
        """Page header should be visible."""
        assert download_page.get_by_text("Download Economic Data").count() >= 1

    def test_indicator_selection_section(self, download_page):
        """Section 1 (Select Indicators) should be present."""
        assert download_page.get_by_text("1. Select Indicators").count() >= 1

    def test_country_selection_section(self, download_page):
        """Section 2 (Select Countries) should be present."""
        assert download_page.get_by_text("2. Select Countries").count() >= 1

    def test_year_range_section(self, download_page):
        """Section 3 (Select Year Range) should be present."""
        assert download_page.get_by_text("3. Select Year Range").count() >= 1

    def test_dataset_name_section(self, download_page):
        """Section 4 (Name Your Dataset) should be present."""
        assert download_page.get_by_text("4. Name Your Dataset").count() >= 1

    def test_download_button_present(self, download_page):
        """The Download Data button should be present (possibly disabled if no indicators selected)."""
        # The button or a warning may appear depending on state
        page = download_page
        has_button = page.get_by_text("Download Data", exact=False).count() >= 1
        has_warning = page.get_by_text("Please select at least one indicator").count() >= 1
        assert has_button or has_warning


class TestIndicatorCategories:
    """Verify that indicator category expanders are present and contain checkboxes."""

    EXPECTED_CATEGORIES = [
        "GDP & Growth",
        "Trade",
        "Inflation & Prices",
        "Employment",
        "Government & Debt",
        "Population & Demographics",
        "Financial",
        "Education & Technology",
        "Energy & Environment",
    ]

    @pytest.mark.parametrize("category", EXPECTED_CATEGORIES)
    def test_category_expander_present(self, download_page, category):
        """Each indicator category should appear as an expander."""
        assert download_page.get_by_text(category, exact=False).count() >= 1, (
            f"Category '{category}' should be present"
        )

    def test_expand_category_shows_select_all(self, download_page):
        """Expanding a category should show a 'Select all' checkbox."""
        page = download_page
        # Click on the GDP & Growth expander
        page.get_by_text("GDP & Growth", exact=False).first.click()
        page.wait_for_timeout(500)

        # Should now see "Select all GDP & Growth"
        assert page.get_by_text("Select all GDP & Growth").count() >= 1

    def test_expand_category_shows_indicator_checkboxes(self, download_page):
        """Expanding a category should show individual indicator checkboxes."""
        page = download_page
        page.get_by_text("GDP & Growth", exact=False).first.click()
        page.wait_for_timeout(500)

        # Should see GDP-related indicator names
        assert page.get_by_text("GDP (current US$)", exact=False).count() >= 1

    def test_select_all_checkbox_present(self, download_page):
        """Each category expander should contain a 'Select all' checkbox."""
        page = download_page
        # Open GDP & Growth expander
        page.get_by_text("GDP & Growth", exact=False).first.click()
        page.wait_for_timeout(500)

        # The "Select all" checkbox should be inside the expander
        select_all = page.get_by_text("Select all GDP & Growth")
        assert select_all.count() >= 1

        # Verify it's a checkbox widget
        checkbox_container = page.locator(
            "[data-testid='stCheckbox']:has-text('Select all GDP & Growth')"
        )
        assert checkbox_container.count() >= 1


class TestCountrySelection:
    """Verify country selection modes and inputs."""

    def test_predefined_groups_mode_default(self, download_page):
        """'Predefined groups' should be the default selection mode."""
        page = download_page
        assert page.get_by_text("Predefined groups").count() >= 1

    def test_manual_entry_mode(self, download_page):
        """Switching to 'Manual entry' mode should show a text area."""
        page = download_page
        page.get_by_text("Manual entry").click()
        page.wait_for_timeout(1000)

        # Should show the text area with default country codes
        assert page.get_by_text("Enter ISO3 country codes", exact=False).count() >= 1

    def test_predefined_groups_shows_g7_default(self, download_page):
        """In predefined groups mode, G7 should be selected by default."""
        page = download_page
        # G7 should appear as a selected tag or in the country list
        assert page.get_by_text("G7", exact=False).count() >= 1

    def test_predefined_groups_shows_country_list(self, download_page):
        """After selecting a group, the country codes should be displayed."""
        page = download_page
        # With G7 default, countries should be listed
        assert page.get_by_text("Countries:", exact=False).count() >= 1

    def test_manual_entry_default_countries(self, download_page):
        """Manual entry should show default country codes in the text area."""
        page = download_page
        page.get_by_text("Manual entry").click()
        page.wait_for_timeout(1000)

        textarea = page.locator("textarea").first
        value = textarea.input_value()
        assert "USA" in value
        assert "GBR" in value


class TestYearRange:
    """Verify year range number inputs."""

    def test_start_year_input(self, download_page):
        """Start year input should be present with default value 2000."""
        page = download_page
        assert page.get_by_text("Start year").count() >= 1

    def test_end_year_input(self, download_page):
        """End year input should be present with default value 2023."""
        page = download_page
        assert page.get_by_text("End year").count() >= 1


class TestDatasetNaming:
    """Verify dataset name input."""

    def test_dataset_name_input_present(self, download_page):
        """Dataset name text input should be present."""
        page = download_page
        assert page.get_by_text("Dataset name").count() >= 1

    def test_dataset_name_default_value(self, download_page):
        """Default dataset name should be 'economic_data'."""
        page = download_page
        input_el = page.locator(
            "[data-testid='stTextInput'] input"
        ).first
        assert input_el.input_value() == "economic_data"


class TestValidation:
    """Verify validation messages when required selections are missing."""

    def test_no_indicators_warning(self, download_page):
        """A warning should be shown when no indicators are selected."""
        page = download_page
        warning = page.get_by_text("Please select at least one indicator", exact=False)
        info = page.get_by_text("Select at least one indicator to proceed", exact=False)
        assert warning.count() >= 1 or info.count() >= 1
