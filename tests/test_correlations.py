"""Tests for analysis/correlations.py module."""

import pytest
import numpy as np
import pandas as pd

from analysis.correlations import (
    pearson_correlation,
    spearman_correlation,
    kendall_correlation,
    partial_correlation,
    correlation_with_pvalues,
    mutual_information_matrix,
    ml_feature_importance_matrix,
    ml_cross_validated_scores,
    pca_analysis,
    CorrelationAutoencoder,
    train_autoencoder,
    granger_causality_test,
    get_top_correlations,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with correlated numeric data."""
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    y = x * 2 + np.random.randn(n) * 0.5  # correlated with x
    z = np.random.randn(n)  # independent
    return pd.DataFrame({"x": x, "y": y, "z": z})


@pytest.fixture
def small_df():
    """Small DataFrame for quick tests."""
    return pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "b": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "c": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    })


@pytest.fixture
def mixed_df():
    """DataFrame with mixed types (should handle non-numeric gracefully)."""
    return pd.DataFrame({
        "country": ["USA", "GBR", "DEU"] * 5,
        "year": list(range(2010, 2025)),
        "gdp": np.random.randn(15) * 1000 + 50000,
        "inflation": np.random.randn(15) * 2 + 3,
    })


@pytest.fixture
def time_series_df():
    """DataFrame for Granger causality tests."""
    np.random.seed(42)
    n = 50
    x = np.cumsum(np.random.randn(n))
    y = np.zeros(n)
    y[0] = np.random.randn()
    for i in range(1, n):
        y[i] = 0.5 * y[i - 1] + 0.3 * x[i - 1] + np.random.randn() * 0.5
    return pd.DataFrame({"x": x, "y": y})


class TestPearsonCorrelation:
    def test_returns_dataframe(self, sample_df):
        result = pearson_correlation(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_diagonal_is_one(self, sample_df):
        result = pearson_correlation(sample_df)
        for col in result.columns:
            assert abs(result.loc[col, col] - 1.0) < 1e-10

    def test_symmetric(self, sample_df):
        result = pearson_correlation(sample_df)
        np.testing.assert_array_almost_equal(result.values, result.values.T)

    def test_high_correlation_detected(self, small_df):
        result = pearson_correlation(small_df)
        # a and b are perfectly correlated
        assert result.loc["a", "b"] > 0.99
        # a and c are perfectly negatively correlated
        assert result.loc["a", "c"] < -0.99

    def test_ignores_non_numeric(self, mixed_df):
        result = pearson_correlation(mixed_df)
        assert "country" not in result.columns

    def test_shape_matches_numeric_columns(self, sample_df):
        result = pearson_correlation(sample_df)
        n = len(sample_df.select_dtypes(include=[np.number]).columns)
        assert result.shape == (n, n)


class TestSpearmanCorrelation:
    def test_returns_dataframe(self, sample_df):
        result = spearman_correlation(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_perfect_monotonic(self, small_df):
        result = spearman_correlation(small_df)
        assert result.loc["a", "b"] > 0.99
        assert result.loc["a", "c"] < -0.99

    def test_symmetric(self, sample_df):
        result = spearman_correlation(sample_df)
        np.testing.assert_array_almost_equal(result.values, result.values.T)


class TestKendallCorrelation:
    def test_returns_dataframe(self, sample_df):
        result = kendall_correlation(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_values_in_range(self, sample_df):
        result = kendall_correlation(sample_df)
        assert (result.values >= -1.0).all()
        assert (result.values <= 1.0).all()


class TestPartialCorrelation:
    def test_returns_dataframe(self, sample_df):
        result = partial_correlation(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_diagonal_is_one(self, sample_df):
        result = partial_correlation(sample_df)
        for col in result.columns:
            assert abs(result.loc[col, col] - 1.0) < 1e-10

    def test_insufficient_data_returns_empty(self):
        # More columns than rows
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
            "d": [7, 8],
        })
        result = partial_correlation(df)
        assert result.empty

    def test_symmetric(self, sample_df):
        result = partial_correlation(sample_df)
        np.testing.assert_array_almost_equal(result.values, result.values.T, decimal=10)


class TestCorrelationWithPvalues:
    def test_returns_two_dataframes(self, sample_df):
        corr, pval = correlation_with_pvalues(sample_df)
        assert isinstance(corr, pd.DataFrame)
        assert isinstance(pval, pd.DataFrame)

    def test_corr_diagonal_is_one(self, sample_df):
        corr, _ = correlation_with_pvalues(sample_df)
        for col in corr.columns:
            assert abs(corr.loc[col, col] - 1.0) < 1e-10

    def test_pval_diagonal_is_zero(self, sample_df):
        _, pval = correlation_with_pvalues(sample_df)
        for col in pval.columns:
            assert abs(pval.loc[col, col]) < 1e-10

    def test_significant_correlation_has_low_pvalue(self, small_df):
        _, pval = correlation_with_pvalues(small_df)
        # a and b are perfectly correlated, p-value should be very small
        assert pval.loc["a", "b"] < 0.01

    def test_spearman_method(self, sample_df):
        corr, pval = correlation_with_pvalues(sample_df, method="spearman")
        assert isinstance(corr, pd.DataFrame)
        assert isinstance(pval, pd.DataFrame)

    def test_kendall_method(self, sample_df):
        corr, pval = correlation_with_pvalues(sample_df, method="kendall")
        assert isinstance(corr, pd.DataFrame)


class TestMutualInformationMatrix:
    def test_returns_dataframe(self, sample_df):
        result = mutual_information_matrix(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_diagonal_is_nan(self, sample_df):
        result = mutual_information_matrix(sample_df)
        for col in result.columns:
            assert np.isnan(result.loc[col, col])

    def test_non_negative(self, sample_df):
        result = mutual_information_matrix(sample_df)
        # Mutual information is non-negative (excluding NaN diagonal)
        mask = ~np.isnan(result.values)
        assert (result.values[mask] >= -0.01).all()  # small tolerance for numerical noise

    def test_correlated_higher_than_independent(self, sample_df):
        result = mutual_information_matrix(sample_df)
        # MI between x and y (correlated) should be higher than x and z (independent)
        mi_xy = result.loc["x", "y"]
        mi_xz = result.loc["x", "z"]
        assert mi_xy > mi_xz


class TestMLFeatureImportance:
    def test_random_forest(self, sample_df):
        result = ml_feature_importance_matrix(sample_df, "random_forest")
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]

    def test_gradient_boosting(self, sample_df):
        result = ml_feature_importance_matrix(sample_df, "gradient_boosting")
        assert isinstance(result, pd.DataFrame)

    def test_lasso(self, sample_df):
        result = ml_feature_importance_matrix(sample_df, "lasso")
        assert isinstance(result, pd.DataFrame)

    def test_elastic_net(self, sample_df):
        result = ml_feature_importance_matrix(sample_df, "elastic_net")
        assert isinstance(result, pd.DataFrame)

    def test_diagonal_is_nan(self, sample_df):
        result = ml_feature_importance_matrix(sample_df, "random_forest")
        for col in result.columns:
            assert np.isnan(result.loc[col, col])

    def test_non_negative_importances(self, sample_df):
        result = ml_feature_importance_matrix(sample_df, "random_forest")
        mask = ~np.isnan(result.values)
        assert (result.values[mask] >= 0).all()


class TestMLCrossValidatedScores:
    def test_returns_dict(self, sample_df):
        result = ml_cross_validated_scores(sample_df, "y")
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, sample_df):
        result = ml_cross_validated_scores(sample_df, "y")
        expected_keys = {"model", "target", "r2_scores", "r2_mean", "r2_std", "n_features", "n_samples"}
        assert expected_keys.issubset(result.keys())

    def test_r2_scores_length_matches_cv(self, sample_df):
        result = ml_cross_validated_scores(sample_df, "y", cv=3)
        assert len(result["r2_scores"]) == 3

    def test_nonexistent_column(self, sample_df):
        result = ml_cross_validated_scores(sample_df, "nonexistent")
        assert "error" in result

    def test_insufficient_data(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = ml_cross_validated_scores(df, "a", cv=5)
        assert "error" in result

    def test_high_r2_for_correlated_data(self, sample_df):
        result = ml_cross_validated_scores(sample_df, "y", model_type="random_forest")
        assert result["r2_mean"] > 0.5  # y is strongly correlated with x


class TestPCAAnalysis:
    def test_returns_dict(self, sample_df):
        result = pca_analysis(sample_df)
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, sample_df):
        result = pca_analysis(sample_df)
        expected = {"explained_variance_ratio", "cumulative_variance", "loadings", "transformed", "n_components"}
        assert expected.issubset(result.keys())

    def test_variance_sums_to_one(self, sample_df):
        result = pca_analysis(sample_df)
        total = sum(result["explained_variance_ratio"])
        assert abs(total - 1.0) < 1e-10

    def test_cumulative_variance_monotonic(self, sample_df):
        result = pca_analysis(sample_df)
        cum_var = result["cumulative_variance"]
        for i in range(1, len(cum_var)):
            assert cum_var[i] >= cum_var[i - 1]

    def test_custom_n_components(self, sample_df):
        result = pca_analysis(sample_df, n_components=2)
        assert result["n_components"] == 2
        assert len(result["explained_variance_ratio"]) == 2

    def test_loadings_shape(self, sample_df):
        result = pca_analysis(sample_df)
        n_features = len(sample_df.columns)
        assert result["loadings"].shape[0] == n_features

    def test_empty_df_returns_error(self):
        df = pd.DataFrame()
        result = pca_analysis(df)
        assert "error" in result


class TestAutoencoder:
    def test_model_creation(self):
        model = CorrelationAutoencoder(input_dim=5, encoding_dim=2)
        assert model is not None

    def test_forward_pass(self):
        import torch
        model = CorrelationAutoencoder(input_dim=5, encoding_dim=2)
        x = torch.randn(10, 5)
        output = model(x)
        assert output.shape == (10, 5)

    def test_encode(self):
        import torch
        model = CorrelationAutoencoder(input_dim=5, encoding_dim=2)
        x = torch.randn(10, 5)
        encoded = model.encode(x)
        assert encoded.shape == (10, 2)

    def test_default_encoding_dim(self):
        import torch
        model = CorrelationAutoencoder(input_dim=9)
        x = torch.randn(10, 9)
        encoded = model.encode(x)
        assert encoded.shape[1] == 3  # max(2, 9 // 3)


class TestTrainAutoencoder:
    def test_returns_dict(self, sample_df):
        result = train_autoencoder(sample_df, epochs=10)
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, sample_df):
        result = train_autoencoder(sample_df, epochs=10)
        expected = {"reconstruction_errors", "importance_matrix", "training_losses", "final_loss"}
        assert expected.issubset(result.keys())

    def test_training_losses_length(self, sample_df):
        result = train_autoencoder(sample_df, epochs=20)
        assert len(result["training_losses"]) == 20

    def test_loss_decreases(self, sample_df):
        result = train_autoencoder(sample_df, epochs=100, lr=0.01)
        losses = result["training_losses"]
        # Overall trend should be decreasing (first loss > last loss)
        assert losses[0] > losses[-1]

    def test_insufficient_data(self):
        df = pd.DataFrame({"a": range(5), "b": range(5)})
        result = train_autoencoder(df)
        assert "error" in result

    def test_reconstruction_errors_shape(self, sample_df):
        result = train_autoencoder(sample_df, epochs=10)
        assert len(result["reconstruction_errors"]) == len(sample_df.columns)

    def test_importance_matrix_shape(self, sample_df):
        result = train_autoencoder(sample_df, epochs=10)
        n = len(sample_df.columns)
        assert result["importance_matrix"].shape == (n, n)


class TestGrangerCausality:
    def test_returns_dict(self, time_series_df):
        result = granger_causality_test(time_series_df, "x", "y", max_lag=2)
        assert isinstance(result, dict)

    def test_contains_lag_results(self, time_series_df):
        result = granger_causality_test(time_series_df, "x", "y", max_lag=2)
        assert "lag_results" in result
        assert "x_causes_y" in result

    def test_lag_results_structure(self, time_series_df):
        result = granger_causality_test(time_series_df, "x", "y", max_lag=2)
        for lag, vals in result["lag_results"].items():
            assert "f_statistic" in vals
            assert "p_value" in vals
            assert "significant_5pct" in vals

    def test_insufficient_data(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = granger_causality_test(df, "x", "y", max_lag=3)
        assert "error" in result

    def test_detects_causality(self, time_series_df):
        # y depends on lagged x by construction, so Granger causality should be detected
        result = granger_causality_test(time_series_df, "x", "y", max_lag=3)
        any_significant = any(
            v["significant_5pct"] for v in result["lag_results"].values()
        )
        assert any_significant, "Expected Granger causality to be detected"


class TestGetTopCorrelations:
    def test_returns_dataframe(self, sample_df):
        corr = pearson_correlation(sample_df)
        result = get_top_correlations(corr)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, sample_df):
        corr = pearson_correlation(sample_df)
        result = get_top_correlations(corr)
        expected = {"indicator_1", "indicator_2", "correlation", "abs_correlation"}
        assert expected.issubset(result.columns)

    def test_respects_n_limit(self, sample_df):
        corr = pearson_correlation(sample_df)
        result = get_top_correlations(corr, n=2)
        assert len(result) <= 2

    def test_sorted_by_abs_correlation(self, sample_df):
        corr = pearson_correlation(sample_df)
        result = get_top_correlations(corr)
        if len(result) > 1:
            values = result["abs_correlation"].values
            assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_no_self_correlations(self, sample_df):
        corr = pearson_correlation(sample_df)
        result = get_top_correlations(corr)
        for _, row in result.iterrows():
            assert row["indicator_1"] != row["indicator_2"]

    def test_min_abs_corr_filter(self, sample_df):
        corr = pearson_correlation(sample_df)
        result = get_top_correlations(corr, min_abs_corr=0.5)
        if not result.empty:
            assert (result["abs_correlation"] >= 0.5).all()

    def test_empty_matrix(self):
        corr = pd.DataFrame()
        result = get_top_correlations(corr)
        assert result.empty
