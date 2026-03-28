"""Tests for the Explore & Visualize page UI elements and interactions."""

import pytest
from tests.e2e.conftest import click_sidebar_nav, TEST_DATASET_NAME

# Plotly charts may render as iframes or divs depending on Streamlit version.
# We check for both .js-plotly-plot class and iframe[title*="streamlit_plotly"]
PLOTLY_SELECTOR = ".js-plotly-plot, iframe[title*='plotly'], [data-testid='stPlotlyChart']"


@pytest.fixture()
def explore_page(app_page):
    """Navigate to the Explore & Visualize page."""
    click_sidebar_nav(app_page, "Explore & Visualize")
    app_page.wait_for_selector("text=Explore & Visualize Data", timeout=10000)
    return app_page


@pytest.fixture()
def explore_page_with_data(explore_page):
    """Navigate to Explore page and load the test dataset from saved files."""
    page = explore_page

    # Switch to "Saved dataset" source
    page.get_by_text("Saved dataset", exact=True).click()
    page.wait_for_timeout(1500)

    # Select the test dataset from the dropdown
    selectbox = page.locator(
        "[data-testid='stSelectbox']:has-text('Select dataset')"
    ).first
    selectbox.click()
    page.wait_for_timeout(300)
    page.get_by_role("option", name=TEST_DATASET_NAME).click()
    page.wait_for_timeout(2000)

    return page


def _wait_for_chart(page, timeout=15000):
    """Wait for a Plotly chart to appear on the page."""
    page.wait_for_selector(PLOTLY_SELECTOR, timeout=timeout)


def _select_chart_type(page, chart_name: str):
    """Select a chart type from the selectbox."""
    page.locator(
        "[data-testid='stSelectbox']:has-text('Chart type')"
    ).first.click()
    page.wait_for_timeout(300)
    page.get_by_role("option", name=chart_name).click()
    page.wait_for_timeout(2000)


class TestExplorePageEmpty:
    """Tests when no data is available in the session."""

    def test_page_header(self, explore_page):
        """Page header should be visible."""
        assert explore_page.get_by_text("Explore & Visualize Data").count() >= 1

    def test_data_source_radio(self, explore_page):
        """Data source radio buttons should be present."""
        page = explore_page
        assert page.get_by_text("Current session").count() >= 1
        assert page.get_by_text("Saved dataset").count() >= 1


class TestExplorePageWithData:
    """Tests with a loaded dataset."""

    def test_chart_type_selectbox_visible(self, explore_page_with_data):
        """Chart type selector should be present once data is loaded."""
        page = explore_page_with_data
        assert page.get_by_text("Chart type").count() >= 1

    def test_visualizations_subheader(self, explore_page_with_data):
        """Visualizations subheader should appear."""
        page = explore_page_with_data
        assert page.get_by_text("Visualizations").count() >= 1

    def test_dataset_overview_expander(self, explore_page_with_data):
        """Dataset Overview expander should be present."""
        page = explore_page_with_data
        assert page.get_by_text("Dataset Overview").count() >= 1

    def test_expand_dataset_overview_shows_metrics(self, explore_page_with_data):
        """Expanding Dataset Overview should show metrics (Rows, Indicators, etc.)."""
        page = explore_page_with_data
        page.get_by_text("Dataset Overview").click()
        page.wait_for_timeout(1000)

        assert page.get_by_text("Rows").count() >= 1
        assert page.get_by_text("Indicators").count() >= 1
        assert page.get_by_text("Countries").count() >= 1
        assert page.get_by_text("Year Range").count() >= 1

    def test_default_chart_is_time_series(self, explore_page_with_data):
        """The default chart type should be Time Series."""
        page = explore_page_with_data
        # Time Series widgets should be visible (Indicator selectbox, Countries multiselect)
        assert page.get_by_text("Indicator").count() >= 1
        assert page.get_by_text("Countries").count() >= 1


class TestTimeSeriesVisualization:
    """Tests for the Time Series chart type."""

    def test_indicator_selectbox(self, explore_page_with_data):
        """Indicator selectbox should be present for Time Series."""
        page = explore_page_with_data
        assert page.get_by_text("Indicator").count() >= 1

    def test_countries_multiselect(self, explore_page_with_data):
        """Countries multiselect should be present for Time Series."""
        page = explore_page_with_data
        assert page.get_by_text("Countries").count() >= 1

    def test_chart_renders(self, explore_page_with_data):
        """A Plotly chart should render for Time Series."""
        page = explore_page_with_data
        _wait_for_chart(page)
        chart = page.locator(PLOTLY_SELECTOR)
        assert chart.count() >= 1


class TestCountryComparisonVisualization:
    """Tests for the Country Comparison (Bar) chart type."""

    @pytest.fixture()
    def bar_page(self, explore_page_with_data):
        """Switch to Country Comparison chart type."""
        page = explore_page_with_data
        _select_chart_type(page, "Country Comparison (Bar)")
        return page

    def test_indicator_selectbox(self, bar_page):
        """Indicator selectbox should be present."""
        assert bar_page.get_by_text("Indicator").count() >= 1

    def test_year_slider(self, bar_page):
        """Year slider should be present."""
        assert bar_page.get_by_text("Year").count() >= 1

    def test_chart_renders(self, bar_page):
        """A bar chart should render."""
        _wait_for_chart(bar_page)
        chart = bar_page.locator(PLOTLY_SELECTOR)
        assert chart.count() >= 1


class TestScatterPlotVisualization:
    """Tests for the Scatter Plot chart type."""

    @pytest.fixture()
    def scatter_page(self, explore_page_with_data):
        """Switch to Scatter Plot chart type."""
        page = explore_page_with_data
        _select_chart_type(page, "Scatter Plot")
        return page

    def test_axis_selectboxes(self, scatter_page):
        """X-axis and Y-axis selectboxes should be present."""
        assert scatter_page.get_by_text("X-axis").count() >= 1
        assert scatter_page.get_by_text("Y-axis").count() >= 1

    def test_color_by_radio(self, scatter_page):
        """Color by radio buttons should be present."""
        assert scatter_page.get_by_text("Color by").count() >= 1
        assert scatter_page.get_by_text("Country", exact=True).count() >= 1

    def test_chart_renders(self, scatter_page):
        """A scatter chart should render."""
        _wait_for_chart(scatter_page)
        chart = scatter_page.locator(PLOTLY_SELECTOR)
        assert chart.count() >= 1


class TestDistributionVisualization:
    """Tests for the Distribution chart type."""

    @pytest.fixture()
    def dist_page(self, explore_page_with_data):
        """Switch to Distribution chart type."""
        page = explore_page_with_data
        _select_chart_type(page, "Distribution")
        return page

    def test_indicator_selectbox(self, dist_page):
        """Indicator selectbox should be present."""
        assert dist_page.get_by_text("Indicator").count() >= 1

    def test_chart_renders(self, dist_page):
        """A histogram should render."""
        _wait_for_chart(dist_page)
        chart = dist_page.locator(PLOTLY_SELECTOR)
        assert chart.count() >= 1

    def test_statistics_metrics(self, dist_page):
        """Mean, Median, and Std Dev metrics should be displayed."""
        assert dist_page.get_by_text("Mean").count() >= 1
        assert dist_page.get_by_text("Median").count() >= 1
        assert dist_page.get_by_text("Std Dev").count() >= 1


class TestHeatmapVisualization:
    """Tests for the Heatmap chart type."""

    @pytest.fixture()
    def heatmap_page(self, explore_page_with_data):
        """Switch to Heatmap chart type."""
        page = explore_page_with_data
        _select_chart_type(page, "Heatmap (by country & year)")
        return page

    def test_indicator_selectbox(self, heatmap_page):
        """Indicator selectbox should be present."""
        assert heatmap_page.get_by_text("Indicator").count() >= 1

    def test_chart_renders(self, heatmap_page):
        """A heatmap should render."""
        _wait_for_chart(heatmap_page)
        chart = heatmap_page.locator(PLOTLY_SELECTOR)
        assert chart.count() >= 1


class TestBoxPlotVisualization:
    """Tests for the Box Plot chart type."""

    @pytest.fixture()
    def boxplot_page(self, explore_page_with_data):
        """Switch to Box Plot chart type."""
        page = explore_page_with_data
        _select_chart_type(page, "Box Plot")
        return page

    def test_indicator_selectbox(self, boxplot_page):
        """Indicator selectbox should be present."""
        assert boxplot_page.get_by_text("Indicator").count() >= 1

    def test_chart_renders(self, boxplot_page):
        """A box plot should render."""
        _wait_for_chart(boxplot_page)
        chart = boxplot_page.locator(PLOTLY_SELECTOR)
        assert chart.count() >= 1
