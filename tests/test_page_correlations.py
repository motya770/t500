"""Tests for ui/page_correlations.py module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture
def corr_df():
    """DataFrame for correlation page tests."""
    np.random.seed(42)
    countries = ["USA", "GBR", "DEU"]
    years = list(range(2015, 2024))
    rows = []
    for country in countries:
        for year in years:
            rows.append({
                "country": country,
                "year": year,
                "NY.GDP.MKTP.CD": np.random.uniform(1e12, 5e12),
                "FP.CPI.TOTL.ZG": np.random.uniform(0, 10),
                "SL.UEM.TOTL.ZS": np.random.uniform(2, 15),
            })
    return pd.DataFrame(rows)


class TestHelperFunctions:
    """Tests for helper functions in page_correlations."""

    @patch("ui.page_correlations.st")
    @patch("ui.page_correlations.get_all_indicators")
    def test_get_indicator_label_from_session(self, mock_all_ind, mock_st):
        mock_st.session_state = {"indicator_names": {"NY.GDP.MKTP.CD": "GDP"}}
        mock_all_ind.return_value = {}

        from ui.page_correlations import _get_indicator_label
        assert _get_indicator_label("NY.GDP.MKTP.CD") == "GDP"

    @patch("ui.page_correlations.st")
    @patch("ui.page_correlations.get_all_indicators")
    def test_get_indicator_label_fallback(self, mock_all_ind, mock_st):
        mock_st.session_state = {}
        mock_all_ind.return_value = {"FP.CPI.TOTL.ZG": "Inflation"}

        from ui.page_correlations import _get_indicator_label
        assert _get_indicator_label("FP.CPI.TOTL.ZG") == "Inflation"

    def test_indicator_columns(self, corr_df):
        from ui.page_correlations import _indicator_columns
        cols = _indicator_columns(corr_df)
        assert "country" not in cols
        assert "year" not in cols
        assert len(cols) == 3


class TestPrepareAnalysisData:
    """Tests for _prepare_analysis_data."""

    def test_pooled_mode(self, corr_df):
        from ui.page_correlations import _prepare_analysis_data
        result = _prepare_analysis_data(corr_df, "All countries (pooled)")
        assert "country" not in result.columns
        assert "year" not in result.columns
        assert len(result) == len(corr_df)

    def test_single_country_mode(self, corr_df):
        from ui.page_correlations import _prepare_analysis_data
        result = _prepare_analysis_data(corr_df, "Single country", "USA")
        assert "country" not in result.columns
        assert len(result) == len(corr_df[corr_df["country"] == "USA"])

    def test_cross_country_average(self, corr_df):
        from ui.page_correlations import _prepare_analysis_data
        result = _prepare_analysis_data(corr_df, "Cross-country average by year")
        assert "country" not in result.columns
        assert "year" not in result.columns
        n_years = corr_df["year"].nunique()
        assert len(result) == n_years

    def test_default_mode(self, corr_df):
        from ui.page_correlations import _prepare_analysis_data
        result = _prepare_analysis_data(corr_df, "unknown mode")
        assert len(result) == len(corr_df)


class TestRenderPage:
    """Tests for the main render() function."""

    @patch("ui.page_correlations.list_saved_datasets")
    @patch("ui.page_correlations.st")
    def test_render_no_data(self, mock_st, mock_list):
        mock_list.return_value = []
        mock_st.session_state = {}

        from ui.page_correlations import render
        render()

        mock_st.info.assert_called()

    @patch("ui.page_correlations.list_saved_datasets")
    @patch("ui.page_correlations.st")
    def test_render_empty_dataset(self, mock_st, mock_list):
        mock_list.return_value = []
        mock_st.session_state = {"current_dataset": pd.DataFrame()}
        mock_st.radio.return_value = "Current session"

        from ui.page_correlations import render
        render()

        mock_st.warning.assert_called()

    @patch("ui.page_correlations.list_saved_datasets")
    @patch("ui.page_correlations.st")
    def test_render_insufficient_indicators(self, mock_st, mock_list):
        """Warns when fewer than 2 indicators available."""
        mock_list.return_value = []
        df = pd.DataFrame({"country": ["USA"], "year": [2020], "ind1": [100]})
        mock_st.session_state = {"current_dataset": df}
        mock_st.radio.return_value = "Current session"

        from ui.page_correlations import render
        render()

        mock_st.warning.assert_called()

    @patch("ui.page_correlations._run_analysis")
    @patch("ui.page_correlations.list_saved_datasets")
    @patch("ui.page_correlations.st")
    def test_render_runs_analysis_on_button_click(self, mock_st, mock_list, mock_run, corr_df):
        mock_list.return_value = []
        mock_st.session_state = {
            "current_dataset": corr_df,
            "indicator_names": {},
        }
        mock_st.radio.return_value = "Current session"

        col_mock = MagicMock()
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [col_mock, col_mock]

        mock_st.selectbox.side_effect = [
            "All countries (pooled)",
            "Drop rows with any NaN",
            "Pearson Correlation",
        ]
        mock_st.button.return_value = True

        from ui.page_correlations import render
        render()

        mock_run.assert_called_once()


class TestRunAnalysis:
    """Tests for _run_analysis dispatcher."""

    @patch("ui.page_correlations._run_correlation")
    @patch("ui.page_correlations.st")
    def test_pearson_dispatches_correctly(self, mock_st, mock_run_corr):
        mock_st.session_state = {}
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        from ui.page_correlations import _run_analysis
        _run_analysis(df, "Pearson Correlation", ["a", "b"], df, [])

        mock_run_corr.assert_called_once_with(df, "pearson")

    @patch("ui.page_correlations._run_correlation")
    @patch("ui.page_correlations.st")
    def test_spearman_dispatches_correctly(self, mock_st, mock_run_corr):
        mock_st.session_state = {}
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        from ui.page_correlations import _run_analysis
        _run_analysis(df, "Spearman Rank Correlation", ["a", "b"], df, [])

        mock_run_corr.assert_called_once_with(df, "spearman")

    @patch("ui.page_correlations._run_correlation")
    @patch("ui.page_correlations.st")
    def test_kendall_dispatches_correctly(self, mock_st, mock_run_corr):
        mock_st.session_state = {}
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        from ui.page_correlations import _run_analysis
        _run_analysis(df, "Kendall Tau Correlation", ["a", "b"], df, [])

        mock_run_corr.assert_called_once_with(df, "kendall")

    @patch("ui.page_correlations._run_partial_correlation")
    @patch("ui.page_correlations.st")
    def test_partial_dispatches_correctly(self, mock_st, mock_run_partial):
        mock_st.session_state = {}
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        from ui.page_correlations import _run_analysis
        _run_analysis(df, "Partial Correlation", ["a", "b"], df, [])

        mock_run_partial.assert_called_once_with(df)

    @patch("ui.page_correlations._run_mutual_information")
    @patch("ui.page_correlations.st")
    def test_mutual_info_dispatches_correctly(self, mock_st, mock_run_mi):
        mock_st.session_state = {}
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        from ui.page_correlations import _run_analysis
        _run_analysis(df, "Mutual Information (non-linear)", ["a", "b"], df, [])

        mock_run_mi.assert_called_once_with(df)

    @patch("ui.page_correlations._run_ml_importance")
    @patch("ui.page_correlations.st")
    def test_ml_rf_dispatches_correctly(self, mock_st, mock_run_ml):
        mock_st.session_state = {}
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        from ui.page_correlations import _run_analysis
        _run_analysis(df, "ML Feature Importance (Random Forest)", ["a", "b"], df, [])

        mock_run_ml.assert_called_once_with(df, "random_forest")

    @patch("ui.page_correlations._run_ml_importance")
    @patch("ui.page_correlations.st")
    def test_ml_gb_dispatches_correctly(self, mock_st, mock_run_ml):
        mock_st.session_state = {}
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        from ui.page_correlations import _run_analysis
        _run_analysis(df, "ML Feature Importance (Gradient Boosting)", ["a", "b"], df, [])

        mock_run_ml.assert_called_once_with(df, "gradient_boosting")

    @patch("ui.page_correlations._run_ml_importance")
    @patch("ui.page_correlations.st")
    def test_ml_lasso_dispatches_correctly(self, mock_st, mock_run_ml):
        mock_st.session_state = {}
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        from ui.page_correlations import _run_analysis
        _run_analysis(df, "ML Feature Importance (Lasso)", ["a", "b"], df, [])

        mock_run_ml.assert_called_once_with(df, "lasso")

    @patch("ui.page_correlations._run_ml_importance")
    @patch("ui.page_correlations.st")
    def test_ml_elastic_net_dispatches_correctly(self, mock_st, mock_run_ml):
        mock_st.session_state = {}
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        from ui.page_correlations import _run_analysis
        _run_analysis(df, "ML Feature Importance (Elastic Net)", ["a", "b"], df, [])

        mock_run_ml.assert_called_once_with(df, "elastic_net")

    @patch("ui.page_correlations._run_pca")
    @patch("ui.page_correlations.st")
    def test_pca_dispatches_correctly(self, mock_st, mock_run_pca):
        mock_st.session_state = {}
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        from ui.page_correlations import _run_analysis
        _run_analysis(df, "PCA Analysis", ["a", "b"], df, [])

        mock_run_pca.assert_called_once_with(df)

    @patch("ui.page_correlations._run_autoencoder")
    @patch("ui.page_correlations.st")
    def test_autoencoder_dispatches_correctly(self, mock_st, mock_run_ae):
        mock_st.session_state = {}
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        from ui.page_correlations import _run_analysis
        _run_analysis(df, "Deep Learning (Autoencoder)", ["a", "b"], df, [])

        mock_run_ae.assert_called_once_with(df)

    @patch("ui.page_correlations._run_granger")
    @patch("ui.page_correlations.st")
    def test_granger_dispatches_correctly(self, mock_st, mock_run_granger):
        mock_st.session_state = {}
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        from ui.page_correlations import _run_analysis
        _run_analysis(df, "Granger Causality Test", ["a", "b"], df, ["USA"])

        mock_run_granger.assert_called_once_with(df, ["a", "b"], ["USA"])

    @patch("ui.page_correlations._run_cv_scores")
    @patch("ui.page_correlations.st")
    def test_cv_scores_dispatches_correctly(self, mock_st, mock_run_cv):
        mock_st.session_state = {}
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        from ui.page_correlations import _run_analysis
        _run_analysis(df, "ML Cross-Validation Scores", ["a", "b"], df, [])

        mock_run_cv.assert_called_once_with(df, ["a", "b"])


class TestRunCorrelation:
    """Tests for _run_correlation."""

    @patch("ui.page_correlations._render_correlation_heatmap")
    @patch("ui.page_correlations.get_top_correlations")
    @patch("ui.page_correlations.correlation_with_pvalues")
    @patch("ui.page_correlations.st")
    def test_run_correlation_calls_methods(self, mock_st, mock_cwp, mock_top, mock_heatmap):
        mock_st.session_state = {"indicator_names": {}}
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

        corr_mat = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]],
            index=["a", "b"], columns=["a", "b"],
        )
        pval_mat = pd.DataFrame(
            [[0.0, 0.001], [0.001, 0.0]],
            index=["a", "b"], columns=["a", "b"],
        )
        mock_cwp.return_value = (corr_mat, pval_mat)
        mock_top.return_value = pd.DataFrame({
            "indicator_1": ["a"],
            "indicator_2": ["b"],
            "correlation": [0.8],
            "abs_correlation": [0.8],
        })

        from ui.page_correlations import _run_correlation
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        _run_correlation(df, "pearson")

        mock_cwp.assert_called_once_with(df, "pearson")
        mock_heatmap.assert_called()


class TestRunPartialCorrelation:
    """Tests for _run_partial_correlation."""

    @patch("ui.page_correlations._render_correlation_heatmap")
    @patch("ui.page_correlations.get_top_correlations")
    @patch("ui.page_correlations.partial_correlation")
    @patch("ui.page_correlations.st")
    def test_empty_result_shows_error(self, mock_st, mock_pcorr, mock_top, mock_heatmap):
        mock_st.session_state = {"indicator_names": {}}
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
        mock_pcorr.return_value = pd.DataFrame()

        from ui.page_correlations import _run_partial_correlation
        _run_partial_correlation(pd.DataFrame())

        mock_st.error.assert_called()

    @patch("ui.page_correlations._render_correlation_heatmap")
    @patch("ui.page_correlations.get_top_correlations")
    @patch("ui.page_correlations.partial_correlation")
    @patch("ui.page_correlations.st")
    def test_success_renders_heatmap(self, mock_st, mock_pcorr, mock_top, mock_heatmap):
        mock_st.session_state = {"indicator_names": {}}
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
        pcorr_mat = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]],
            index=["a", "b"], columns=["a", "b"],
        )
        mock_pcorr.return_value = pcorr_mat
        mock_top.return_value = pd.DataFrame({
            "indicator_1": ["a"],
            "indicator_2": ["b"],
            "correlation": [0.5],
            "abs_correlation": [0.5],
        })

        from ui.page_correlations import _run_partial_correlation
        _run_partial_correlation(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

        mock_heatmap.assert_called()


class TestRunPCA:
    """Tests for _run_pca."""

    @patch("ui.page_correlations.apply_steam_style")
    @patch("ui.page_correlations.px")
    @patch("ui.page_correlations.go")
    @patch("ui.page_correlations.pca_analysis")
    @patch("ui.page_correlations.st")
    def test_pca_error(self, mock_st, mock_pca, mock_go, mock_px, mock_style):
        mock_st.session_state = {"indicator_names": {}}
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
        mock_pca.return_value = {"error": "No numeric data"}

        from ui.page_correlations import _run_pca
        _run_pca(pd.DataFrame())

        mock_st.error.assert_called_with("No numeric data")

    @patch("ui.page_correlations.apply_steam_style")
    @patch("ui.page_correlations.px")
    @patch("ui.page_correlations.go")
    @patch("ui.page_correlations.pca_analysis")
    @patch("ui.page_correlations.st")
    def test_pca_success(self, mock_st, mock_pca, mock_go, mock_px, mock_style):
        mock_st.session_state = {"indicator_names": {"a": "A", "b": "B"}}
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        loadings = pd.DataFrame(
            [[0.5, 0.5], [0.5, -0.5]],
            index=["a", "b"],
            columns=["PC1", "PC2"],
        )
        transformed = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]],
            columns=["PC1", "PC2"],
        )
        mock_pca.return_value = {
            "explained_variance_ratio": [0.7, 0.3],
            "cumulative_variance": [0.7, 1.0],
            "loadings": loadings,
            "transformed": transformed,
            "n_components": 2,
        }

        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig
        mock_px.imshow.return_value = MagicMock()
        mock_px.scatter.return_value = MagicMock()

        from ui.page_correlations import _run_pca
        _run_pca(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))

        mock_pca.assert_called_once()
        assert mock_st.plotly_chart.called


class TestRunGranger:
    """Tests for _run_granger."""

    @patch("ui.page_correlations.granger_causality_test")
    @patch("ui.page_correlations.st")
    def test_granger_error(self, mock_st, mock_granger):
        mock_st.session_state = {"indicator_names": {}}
        col_mock = MagicMock()
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [col_mock, col_mock]
        mock_st.selectbox.side_effect = ["a", "b", "USA"]
        mock_st.slider.return_value = 3
        mock_st.button.return_value = True
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        mock_granger.return_value = {"error": "Not enough data"}

        from ui.page_correlations import _run_granger
        df = pd.DataFrame({"country": ["USA"], "year": [2020], "a": [1], "b": [2]})
        _run_granger(df, ["a", "b"], ["USA"])

        mock_st.error.assert_called()

    @patch("ui.page_correlations.granger_causality_test")
    @patch("ui.page_correlations.st")
    def test_granger_success(self, mock_st, mock_granger):
        mock_st.session_state = {"indicator_names": {"a": "Ind A", "b": "Ind B"}}
        col_mock = MagicMock()
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [col_mock, col_mock]
        mock_st.selectbox.side_effect = ["a", "b", "USA"]
        mock_st.slider.return_value = 2
        mock_st.button.return_value = True
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        mock_granger.return_value = {
            "x_causes_y": "a -> b",
            "lag_results": {
                1: {"f_statistic": 5.0, "p_value": 0.03, "significant_5pct": True},
                2: {"f_statistic": 3.0, "p_value": 0.08, "significant_5pct": False},
            },
        }

        from ui.page_correlations import _run_granger
        df = pd.DataFrame({
            "country": ["USA"] * 10,
            "year": list(range(2010, 2020)),
            "a": list(range(10)),
            "b": list(range(10)),
        })
        _run_granger(df, ["a", "b"], ["USA"])

        mock_st.success.assert_called()


class TestRunCVScores:
    """Tests for _run_cv_scores."""

    @patch("ui.page_correlations.apply_steam_style")
    @patch("ui.page_correlations.px")
    @patch("ui.page_correlations.ml_cross_validated_scores")
    @patch("ui.page_correlations.st")
    def test_cv_error(self, mock_st, mock_cv, mock_px, mock_style):
        mock_st.session_state = {"indicator_names": {}}
        mock_st.selectbox.side_effect = ["a", "random_forest"]
        mock_st.slider.return_value = 5
        mock_st.button.return_value = True
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        mock_cv.return_value = {"error": "Not enough data"}

        from ui.page_correlations import _run_cv_scores
        _run_cv_scores(pd.DataFrame({"a": [1], "b": [2]}), ["a", "b"])

        mock_st.error.assert_called()

    @patch("ui.page_correlations.apply_steam_style")
    @patch("ui.page_correlations.px")
    @patch("ui.page_correlations.ml_cross_validated_scores")
    @patch("ui.page_correlations.st")
    def test_cv_high_r2(self, mock_st, mock_cv, mock_px, mock_style):
        mock_st.session_state = {"indicator_names": {"a": "Ind A"}}
        mock_st.selectbox.side_effect = ["a", "random_forest"]
        mock_st.slider.return_value = 5
        mock_st.button.return_value = True
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        mock_cols = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.columns.return_value = mock_cols

        mock_fig = MagicMock()
        mock_px.bar.return_value = mock_fig

        mock_cv.return_value = {
            "model": "random_forest",
            "target": "a",
            "r2_scores": [0.8, 0.85, 0.82, 0.78, 0.9],
            "r2_mean": 0.83,
            "r2_std": 0.04,
            "n_features": 1,
            "n_samples": 50,
        }

        from ui.page_correlations import _run_cv_scores
        _run_cv_scores(pd.DataFrame({"a": range(50), "b": range(50)}), ["a", "b"])

        mock_st.success.assert_called()

    @patch("ui.page_correlations.apply_steam_style")
    @patch("ui.page_correlations.px")
    @patch("ui.page_correlations.ml_cross_validated_scores")
    @patch("ui.page_correlations.st")
    def test_cv_low_r2(self, mock_st, mock_cv, mock_px, mock_style):
        mock_st.session_state = {"indicator_names": {"a": "Ind A"}}
        mock_st.selectbox.side_effect = ["a", "random_forest"]
        mock_st.slider.return_value = 5
        mock_st.button.return_value = True
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        mock_cols = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.columns.return_value = mock_cols

        mock_fig = MagicMock()
        mock_px.bar.return_value = mock_fig

        mock_cv.return_value = {
            "model": "random_forest",
            "target": "a",
            "r2_scores": [0.1, 0.05, 0.2, 0.08, 0.15],
            "r2_mean": 0.12,
            "r2_std": 0.06,
            "n_features": 1,
            "n_samples": 50,
        }

        from ui.page_correlations import _run_cv_scores
        _run_cv_scores(pd.DataFrame({"a": range(50), "b": range(50)}), ["a", "b"])

        mock_st.warning.assert_called()


class TestRenderCorrelationHeatmap:
    """Tests for _render_correlation_heatmap."""

    @patch("ui.page_correlations.apply_steam_style")
    @patch("ui.page_correlations.go")
    @patch("ui.page_correlations.st")
    def test_renders_heatmap(self, mock_st, mock_go, mock_style):
        mock_st.session_state = {"indicator_names": {"a": "A", "b": "B"}}
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        matrix = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]],
            index=["a", "b"], columns=["a", "b"],
        )

        from ui.page_correlations import _render_correlation_heatmap
        _render_correlation_heatmap(matrix, "Test Heatmap")

        mock_go.Figure.assert_called_once()
        mock_st.plotly_chart.assert_called_once()
