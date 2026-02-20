"""Correlation analysis module.

Provides classical statistical and ML/DL methods for finding correlations
between economic indicators.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_regression
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Classical correlation methods
# ---------------------------------------------------------------------------

def pearson_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    return numeric.corr(method="pearson")


def spearman_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman rank correlation matrix."""
    numeric = df.select_dtypes(include=[np.number])
    return numeric.corr(method="spearman")


def kendall_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Kendall tau correlation matrix."""
    numeric = df.select_dtypes(include=[np.number])
    return numeric.corr(method="kendall")


def partial_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute partial correlation matrix (controlling for other variables)."""
    numeric = df.select_dtypes(include=[np.number]).dropna()
    if numeric.shape[0] < numeric.shape[1] + 1:
        return pd.DataFrame()

    corr = numeric.corr().values
    try:
        precision = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(corr)

    diag = np.sqrt(np.diag(precision))
    diag_outer = np.outer(diag, diag)
    diag_outer[diag_outer == 0] = 1
    partial = -precision / diag_outer
    np.fill_diagonal(partial, 1.0)

    cols = numeric.columns
    return pd.DataFrame(partial, index=cols, columns=cols)


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def correlation_with_pvalues(
    df: pd.DataFrame, method: str = "pearson"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute correlation matrix with p-values.

    Returns (correlation_matrix, pvalue_matrix).
    """
    numeric = df.select_dtypes(include=[np.number]).dropna()
    cols = numeric.columns
    n = len(cols)
    corr_mat = np.zeros((n, n))
    pval_mat = np.zeros((n, n))

    corr_func = {
        "pearson": stats.pearsonr,
        "spearman": stats.spearmanr,
        "kendall": stats.kendalltau,
    }[method]

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_mat[i, j] = 1.0
                pval_mat[i, j] = 0.0
            elif i < j:
                r, p = corr_func(numeric.iloc[:, i], numeric.iloc[:, j])
                corr_mat[i, j] = r
                corr_mat[j, i] = r
                pval_mat[i, j] = p
                pval_mat[j, i] = p

    return (
        pd.DataFrame(corr_mat, index=cols, columns=cols),
        pd.DataFrame(pval_mat, index=cols, columns=cols),
    )


# ---------------------------------------------------------------------------
# Mutual information (non-linear dependencies)
# ---------------------------------------------------------------------------

def mutual_information_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise mutual information between numeric columns.

    Captures non-linear dependencies that correlation misses.
    """
    numeric = df.select_dtypes(include=[np.number]).dropna()
    cols = numeric.columns
    n = len(cols)
    mi_mat = np.zeros((n, n))

    X = numeric.values
    for i in range(n):
        mi_scores = mutual_info_regression(
            np.delete(X, i, axis=1),
            X[:, i],
            random_state=42,
        )
        idx = 0
        for j in range(n):
            if j == i:
                mi_mat[i, j] = np.nan
            else:
                mi_mat[i, j] = mi_scores[idx]
                idx += 1

    # Symmetrize
    mi_symmetric = (mi_mat + mi_mat.T) / 2
    np.fill_diagonal(mi_symmetric, np.nan)

    return pd.DataFrame(mi_symmetric, index=cols, columns=cols)


# ---------------------------------------------------------------------------
# ML-based feature importance (Random Forest, Gradient Boosting, Lasso)
# ---------------------------------------------------------------------------

def _feature_importance_for_target(
    df: pd.DataFrame, target_col: str, model_type: str = "random_forest"
) -> pd.Series:
    """Compute feature importances for predicting a target column."""
    numeric = df.select_dtypes(include=[np.number]).dropna()
    if target_col not in numeric.columns:
        return pd.Series(dtype=float)

    X = numeric.drop(columns=[target_col])
    y = numeric[target_col]

    if len(X) < 10 or X.shape[1] == 0:
        return pd.Series(0.0, index=X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_scaled, y)
        importances = model.feature_importances_
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        importances = model.feature_importances_
    elif model_type == "lasso":
        model = Lasso(alpha=0.1, random_state=42, max_iter=10000)
        model.fit(X_scaled, y)
        importances = np.abs(model.coef_)
    elif model_type == "elastic_net":
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000)
        model.fit(X_scaled, y)
        importances = np.abs(model.coef_)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return pd.Series(importances, index=X.columns)


def ml_feature_importance_matrix(
    df: pd.DataFrame, model_type: str = "random_forest"
) -> pd.DataFrame:
    """Compute feature importance matrix using ML models.

    Each row i, column j shows how important indicator j is
    for predicting indicator i.
    """
    numeric = df.select_dtypes(include=[np.number]).dropna()
    cols = numeric.columns
    importance_data = {}

    for col in cols:
        importances = _feature_importance_for_target(numeric, col, model_type)
        row = {}
        for c in cols:
            if c == col:
                row[c] = np.nan
            elif c in importances.index:
                row[c] = importances[c]
            else:
                row[c] = 0.0
        importance_data[col] = row

    return pd.DataFrame(importance_data).T


def ml_cross_validated_scores(
    df: pd.DataFrame, target_col: str, model_type: str = "random_forest", cv: int = 5
) -> dict:
    """Run cross-validated scoring for a target indicator.

    Returns dict with model info and R² scores.
    """
    numeric = df.select_dtypes(include=[np.number]).dropna()
    if target_col not in numeric.columns:
        return {"error": f"Column {target_col} not found"}

    X = numeric.drop(columns=[target_col])
    y = numeric[target_col]

    if len(X) < cv * 2:
        return {"error": "Not enough data for cross-validation"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "lasso": Lasso(alpha=0.1, random_state=42),
        "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    }

    if model_type not in models:
        return {"error": f"Unknown model type: {model_type}"}

    model = models[model_type]
    actual_cv = min(cv, len(X))
    scores = cross_val_score(model, X_scaled, y, cv=actual_cv, scoring="r2")

    return {
        "model": model_type,
        "target": target_col,
        "r2_scores": scores.tolist(),
        "r2_mean": float(scores.mean()),
        "r2_std": float(scores.std()),
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
    }


# ---------------------------------------------------------------------------
# PCA-based analysis
# ---------------------------------------------------------------------------

def pca_analysis(df: pd.DataFrame, n_components: int = None) -> dict:
    """Run PCA on the data and return explained variance and loadings."""
    numeric = df.select_dtypes(include=[np.number]).dropna()
    if numeric.empty:
        return {"error": "No numeric data"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric)

    if n_components is None:
        n_components = min(numeric.shape)

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=numeric.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "loadings": loadings,
        "transformed": pd.DataFrame(
            transformed,
            columns=[f"PC{i+1}" for i in range(n_components)],
        ),
        "n_components": n_components,
    }


# ---------------------------------------------------------------------------
# Deep Learning: Autoencoder for non-linear correlation discovery
# ---------------------------------------------------------------------------

class CorrelationAutoencoder(nn.Module):
    """Autoencoder for discovering non-linear relationships between indicators."""

    def __init__(self, input_dim: int, encoding_dim: int = None):
        super().__init__()
        if encoding_dim is None:
            encoding_dim = max(2, input_dim // 3)

        mid_dim = max(encoding_dim + 1, (input_dim + encoding_dim) // 2)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mid_dim),
            nn.Linear(mid_dim, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, mid_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mid_dim),
            nn.Linear(mid_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(
    df: pd.DataFrame,
    encoding_dim: int = None,
    epochs: int = 200,
    lr: float = 0.001,
    progress_callback=None,
) -> dict:
    """Train an autoencoder to find non-linear structure in the data.

    Returns reconstruction errors per feature (high error = less predictable
    from other features = more independent).
    """
    numeric = df.select_dtypes(include=[np.number]).dropna()
    if numeric.shape[0] < 20:
        return {"error": "Not enough data (need at least 20 rows)"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric)
    X_tensor = torch.FloatTensor(X_scaled)

    input_dim = X_scaled.shape[1]
    model = CorrelationAutoencoder(input_dim, encoding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="none")

    losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, X_tensor).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if progress_callback and epoch % 10 == 0:
            progress_callback(epoch, epochs, loss.item())

    # Per-feature reconstruction error
    model.eval()
    with torch.no_grad():
        output = model(X_tensor)
        per_feature_error = criterion(output, X_tensor).mean(dim=0).numpy()

    reconstruction_errors = pd.Series(per_feature_error, index=numeric.columns)

    # Feature importance via perturbation
    importance_matrix = np.zeros((input_dim, input_dim))
    with torch.no_grad():
        baseline_output = model(X_tensor)
        baseline_errors = criterion(baseline_output, X_tensor).mean(dim=0).numpy()

        for i in range(input_dim):
            perturbed = X_tensor.clone()
            perturbed[:, i] = torch.randn(X_tensor.shape[0])
            perturbed_output = model(perturbed)
            perturbed_errors = criterion(perturbed_output, X_tensor).mean(dim=0).numpy()
            importance_matrix[:, i] = perturbed_errors - baseline_errors

    importance_df = pd.DataFrame(
        importance_matrix,
        index=numeric.columns,
        columns=numeric.columns,
    )

    return {
        "reconstruction_errors": reconstruction_errors,
        "importance_matrix": importance_df,
        "training_losses": losses,
        "final_loss": losses[-1],
    }


# ---------------------------------------------------------------------------
# Granger causality (time-series specific)
# ---------------------------------------------------------------------------

def granger_causality_test(
    df: pd.DataFrame, col_x: str, col_y: str, max_lag: int = 3
) -> dict:
    """Test if col_x Granger-causes col_y (for time-series panel data).

    Runs a simple F-test comparing AR model with and without lagged values of col_x.
    """
    data = df[[col_x, col_y]].dropna()
    if len(data) < max_lag * 3:
        return {"error": "Not enough data for Granger causality test"}

    y = data[col_y].values
    x = data[col_x].values

    results = {}
    for lag in range(1, max_lag + 1):
        n = len(y) - lag
        if n < lag + 2:
            continue

        # Restricted model: y_t ~ y_{t-1}, ..., y_{t-lag}
        Y = y[lag:]
        X_restricted = np.column_stack([y[lag - k: -k if k > 0 else len(y)] for k in range(1, lag + 1)])

        # Unrestricted: add lagged x
        X_unrestricted = np.column_stack([
            X_restricted,
            *[x[lag - k: -k if k > 0 else len(x)] for k in range(1, lag + 1)]
        ])

        # Add intercept
        X_r = np.column_stack([np.ones(len(Y)), X_restricted])
        X_u = np.column_stack([np.ones(len(Y)), X_unrestricted])

        try:
            # OLS
            beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]

            resid_r = Y - X_r @ beta_r
            resid_u = Y - X_u @ beta_u

            ssr_r = np.sum(resid_r ** 2)
            ssr_u = np.sum(resid_u ** 2)

            df_diff = lag
            df_resid = len(Y) - X_u.shape[1]

            if ssr_u > 0 and df_resid > 0:
                f_stat = ((ssr_r - ssr_u) / df_diff) / (ssr_u / df_resid)
                p_value = 1 - stats.f.cdf(f_stat, df_diff, df_resid)
                results[lag] = {
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value),
                    "significant_5pct": p_value < 0.05,
                }
        except Exception:
            continue

    return {
        "x_causes_y": f"{col_x} -> {col_y}",
        "lag_results": results,
    }


# ---------------------------------------------------------------------------
# Top correlations extraction
# ---------------------------------------------------------------------------

def get_top_correlations(
    corr_matrix: pd.DataFrame, n: int = 20, min_abs_corr: float = 0.0
) -> pd.DataFrame:
    """Extract top N strongest correlations from a correlation matrix."""
    # Get upper triangle only (avoid duplicates)
    mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    pairs = []

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if mask[i, j] and not np.isnan(corr_matrix.iloc[i, j]):
                val = corr_matrix.iloc[i, j]
                if abs(val) >= min_abs_corr:
                    pairs.append({
                        "indicator_1": corr_matrix.index[i],
                        "indicator_2": corr_matrix.columns[j],
                        "correlation": val,
                        "abs_correlation": abs(val),
                    })

    result = pd.DataFrame(pairs)
    if result.empty:
        return result

    result = result.sort_values("abs_correlation", ascending=False).head(n)
    return result.reset_index(drop=True)
