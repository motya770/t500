"""Tests for the Correlation Analysis page UI elements and interactions."""

import pytest
from tests.e2e.conftest import click_sidebar_nav, TEST_DATASET_NAME

PLOTLY_SELECTOR = ".js-plotly-plot, iframe[title*='plotly'], [data-testid='stPlotlyChart']"

ANALYSIS_METHODS = [
    "Pearson Correlation",
    "Spearman Rank Correlation",
    "Kendall Tau Correlation",
    "Partial Correlation",
    "Mutual Information (non-linear)",
    "ML Feature Importance (Random Forest)",
    "ML Feature Importance (Gradient Boosting)",
    "ML Feature Importance (Lasso)",
    "ML Feature Importance (Elastic Net)",
    "PCA Analysis",
    "Deep Learning (Autoencoder)",
    "Granger Causality Test",
    "ML Cross-Validation Scores",
]


def _select_analysis_method(page, method_name: str) -> None:
    """Select an analysis method from the dropdown, scrolling if needed.

    Streamlit selectbox dropdowns virtualise options; items near the bottom
    may not exist in the DOM until the user scrolls.  We open the dropdown,
    then type a search prefix to filter the list to the desired option.
    """
    selectbox = page.locator(
        "[data-testid='stSelectbox']:has-text('Choose analysis method')"
    ).first
    selectbox.click()
    page.wait_for_timeout(300)

    option = page.get_by_role("option", name=method_name)
    if option.count() > 0:
        option.click()
    else:
        # Type the first unique word(s) of the method name to filter
        # Streamlit selectbox supports keyboard search when open
        search_text = method_name.split("(")[0].strip()[:20]
        page.keyboard.type(search_text, delay=50)
        page.wait_for_timeout(500)
        page.get_by_role("option", name=method_name).first.click()
    page.wait_for_timeout(1000)


@pytest.fixture()
def correlations_page(app_page):
    """Navigate to the Correlation Analysis page."""
    click_sidebar_nav(app_page, "Correlation Analysis")
    app_page.wait_for_selector("text=Correlation & ML Analysis", timeout=30000)
    return app_page


@pytest.fixture()
def correlations_page_with_data(correlations_page):
    """Navigate to Correlations page and load the test dataset."""
    page = correlations_page

    # Switch to "Saved dataset" source
    page.get_by_text("Saved dataset", exact=True).click()
    page.wait_for_timeout(1500)

    # Select the test dataset
    selectbox = page.locator(
        "[data-testid='stSelectbox']:has-text('Select dataset')"
    ).first
    selectbox.click()
    page.wait_for_timeout(300)
    page.get_by_role("option", name=TEST_DATASET_NAME).click()
    page.wait_for_timeout(2000)

    return page


class TestCorrelationsPageLayout:
    """Verify that all expected sections and widgets render."""

    def test_page_header(self, correlations_page):
        """Page header should be visible."""
        assert correlations_page.get_by_text("Correlation & ML Analysis").count() >= 1

    def test_data_source_radio(self, correlations_page):
        """Data source radio buttons should be present."""
        page = correlations_page
        assert page.get_by_text("Current session").count() >= 1
        assert page.get_by_text("Saved dataset").count() >= 1


class TestCorrelationsDataPreparation:
    """Tests for the data preparation section with a loaded dataset."""

    def test_data_preparation_section(self, correlations_page_with_data):
        """Data Preparation subheader should appear."""
        page = correlations_page_with_data
        assert page.get_by_text("Data Preparation").count() >= 1

    def test_aggregation_mode_selectbox(self, correlations_page_with_data):
        """Aggregation mode selectbox should be present."""
        page = correlations_page_with_data
        assert page.get_by_text("Aggregation mode").count() >= 1

    def test_aggregation_mode_options(self, correlations_page_with_data):
        """Aggregation mode selectbox should have the expected options."""
        page = correlations_page_with_data

        page.locator(
            "[data-testid='stSelectbox']:has-text('Aggregation mode')"
        ).first.click()
        page.wait_for_timeout(300)

        assert page.get_by_role("option", name="All countries (pooled)").count() >= 1
        assert page.get_by_role("option", name="Single country").count() >= 1
        assert page.get_by_role("option", name="Cross-country average by year").count() >= 1

        page.keyboard.press("Escape")
        page.wait_for_timeout(300)

    def test_single_country_shows_country_selector(self, correlations_page_with_data):
        """Selecting 'Single country' mode should show a Country selectbox."""
        page = correlations_page_with_data

        page.locator(
            "[data-testid='stSelectbox']:has-text('Aggregation mode')"
        ).first.click()
        page.wait_for_timeout(300)
        page.get_by_role("option", name="Single country").click()
        page.wait_for_timeout(1500)

        assert page.get_by_text("Country", exact=True).count() >= 1

    def test_missing_values_selectbox(self, correlations_page_with_data):
        """Handle missing values selectbox should be present."""
        page = correlations_page_with_data
        assert page.get_by_text("Handle missing values").count() >= 1

    def test_missing_values_options(self, correlations_page_with_data):
        """Missing values selectbox should have the expected options."""
        page = correlations_page_with_data

        page.locator(
            "[data-testid='stSelectbox']:has-text('Handle missing values')"
        ).first.click()
        page.wait_for_timeout(300)

        assert page.get_by_role("option", name="Drop rows with any NaN").count() >= 1
        assert page.get_by_role("option", name="Forward fill then drop").count() >= 1
        assert page.get_by_role("option", name="Mean imputation").count() >= 1

        page.keyboard.press("Escape")
        page.wait_for_timeout(300)

    def test_analysis_data_info(self, correlations_page_with_data):
        """Data summary showing rows and indicators should appear."""
        page = correlations_page_with_data
        assert page.get_by_text("Analysis data:", exact=False).count() >= 1
        assert page.get_by_text("rows", exact=False).count() >= 1
        assert page.get_by_text("indicators", exact=False).count() >= 1


class TestAnalysisMethodSelection:
    """Tests for the analysis method selection section."""

    def test_analysis_methods_subheader(self, correlations_page_with_data):
        """Analysis Methods subheader should appear."""
        page = correlations_page_with_data
        assert page.get_by_text("Analysis Methods").count() >= 1

    def test_method_selectbox_present(self, correlations_page_with_data):
        """Analysis method selectbox should be present."""
        page = correlations_page_with_data
        assert page.get_by_text("Choose analysis method").count() >= 1

    def test_run_analysis_button(self, correlations_page_with_data):
        """Run Analysis button should be present."""
        page = correlations_page_with_data
        run_btn = page.get_by_text("Run Analysis", exact=False)
        assert run_btn.count() >= 1

    def test_method_selectbox_has_core_methods(self, correlations_page_with_data):
        """Analysis method dropdown should contain the core methods.

        The Streamlit selectbox virtualizes its option list, so only
        visible options are in the DOM. We verify the first few items.
        """
        page = correlations_page_with_data

        page.locator(
            "[data-testid='stSelectbox']:has-text('Choose analysis method')"
        ).first.click()
        page.wait_for_timeout(300)

        visible_methods = [
            "Pearson Correlation",
            "Spearman Rank Correlation",
            "Kendall Tau Correlation",
            "Partial Correlation",
            "Mutual Information (non-linear)",
        ]
        for method in visible_methods:
            assert page.get_by_role("option", name=method).count() >= 1, (
                f"Method '{method}' should be in dropdown"
            )

        page.keyboard.press("Escape")
        page.wait_for_timeout(300)


class TestPearsonCorrelation:
    """Test running Pearson Correlation analysis end-to-end."""

    def test_pearson_produces_heatmap(self, correlations_page_with_data):
        """Running Pearson correlation should produce a heatmap and top correlations."""
        page = correlations_page_with_data

        page.locator("button:has-text('Run Analysis')").click()
        page.wait_for_selector("text=Strongest Correlations", timeout=15000)
        # Wait for chart to render after text appears
        page.wait_for_selector(PLOTLY_SELECTOR, timeout=10000)

        chart = page.locator(PLOTLY_SELECTOR)
        assert chart.count() >= 1, "Heatmap should render after running Pearson"
        assert page.get_by_text("Strongest Correlations").count() >= 1

    def test_pearson_shows_pvalues_expander(self, correlations_page_with_data):
        """Running Pearson should show a P-values expander."""
        page = correlations_page_with_data

        page.locator("button:has-text('Run Analysis')").click()
        page.wait_for_selector("text=P-values", timeout=15000)

        assert page.get_by_text("P-values", exact=False).count() >= 1


class TestSpearmanCorrelation:
    """Test running Spearman correlation analysis."""

    def test_spearman_produces_results(self, correlations_page_with_data):
        """Running Spearman correlation should produce a heatmap."""
        page = correlations_page_with_data

        _select_analysis_method(page, "Spearman Rank Correlation")

        page.locator("button:has-text('Run Analysis')").click()
        page.wait_for_selector("text=Strongest Correlations", timeout=15000)
        page.wait_for_selector(PLOTLY_SELECTOR, timeout=10000)

        chart = page.locator(PLOTLY_SELECTOR)
        assert chart.count() >= 1
        assert page.get_by_text("Strongest Correlations").count() >= 1


class TestPCAAnalysis:
    """Test running PCA analysis."""

    def test_pca_produces_results(self, correlations_page_with_data):
        """Running PCA should produce variance and loadings charts."""
        page = correlations_page_with_data

        _select_analysis_method(page, "PCA Analysis")

        page.locator("button:has-text('Run Analysis')").click()
        page.wait_for_selector("text=Explained Variance", timeout=15000)
        page.wait_for_selector(PLOTLY_SELECTOR, timeout=10000)

        assert page.get_by_text("Explained Variance").count() >= 1
        assert page.get_by_text("Component Loadings").count() >= 1

        chart = page.locator(PLOTLY_SELECTOR)
        assert chart.count() >= 1


class TestGrangerCausality:
    """Test that Granger Causality UI elements render correctly."""

    @pytest.fixture()
    def granger_page(self, correlations_page_with_data):
        """Switch to Granger Causality method and run analysis."""
        page = correlations_page_with_data
        _select_analysis_method(page, "Granger Causality Test")

        page.locator("button:has-text('Run Analysis')").click()
        # Wait for the Granger UI to appear
        page.wait_for_selector("text=Cause (X)", timeout=15000)

        return page

    def test_granger_cause_effect_selectors(self, granger_page):
        """Granger test should show Cause (X) and Effect (Y) selectors."""
        page = granger_page
        assert page.get_by_text("Cause (X)").count() >= 1
        assert page.get_by_text("Effect (Y)").count() >= 1

    def test_granger_country_selector(self, granger_page):
        """Granger test should show a Country selector."""
        page = granger_page
        assert page.get_by_text("Country").count() >= 1

    def test_granger_max_lag_slider(self, granger_page):
        """Granger test should show a max lag slider."""
        page = granger_page
        # The slider label is "Max lag (years)" — check for either form
        has_label = (
            page.get_by_text("Max lag", exact=False).count() >= 1
            or page.locator("[data-testid='stSlider']").count() >= 1
        )
        assert has_label

    def test_granger_run_button(self, granger_page):
        """Granger test should show a 'Run Granger Test' button."""
        page = granger_page
        assert page.get_by_text("Run Granger Test").count() >= 1


class TestCrossValidation:
    """Test that Cross-Validation UI elements render correctly."""

    @pytest.fixture()
    def cv_page(self, correlations_page_with_data):
        """Switch to ML Cross-Validation method and run analysis."""
        page = correlations_page_with_data
        _select_analysis_method(page, "ML Cross-Validation Scores")

        page.locator("button:has-text('Run Analysis')").click()
        # Wait for the CV UI to appear
        page.wait_for_selector("text=Target indicator", timeout=15000)

        return page

    def test_cv_target_selector(self, cv_page):
        """CV should show Target indicator selector."""
        page = cv_page
        assert page.get_by_text("Target indicator").count() >= 1

    def test_cv_model_selector(self, cv_page):
        """CV should show Model selector."""
        page = cv_page
        assert page.get_by_text("Model", exact=True).count() >= 1

    def test_cv_folds_slider(self, cv_page):
        """CV should show CV folds slider."""
        page = cv_page
        assert page.get_by_text("CV folds").count() >= 1

    def test_cv_run_button(self, cv_page):
        """CV should show a 'Run Cross-Validation' button."""
        page = cv_page
        assert page.get_by_text("Run Cross-Validation").count() >= 1
