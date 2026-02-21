"""Correlation Analysis page for the Streamlit app."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_sources.world_bank import list_saved_datasets, load_dataset, get_all_indicators
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
    train_autoencoder,
    granger_causality_test,
    get_top_correlations,
)
from ui.theme import (
    apply_steam_style, CHART_COLORS, DIVERGING_SCALE, HEATMAP_SCALE,
    BRASS, COPPER, CREAM, EMBER, COAL, DARK_IRON, DARK_WOOD, BRONZE,
)


def _get_indicator_label(code: str) -> str:
    names = st.session_state.get("indicator_names", {})
    if code in names:
        return names[code]
    all_ind = get_all_indicators()
    return all_ind.get(code, code)


def _indicator_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("country", "year")]


def _prepare_analysis_data(df: pd.DataFrame, mode: str, country: str = None) -> pd.DataFrame:
    """Prepare data for analysis depending on the aggregation mode."""
    indicators = _indicator_columns(df)
    if mode == "All countries (pooled)":
        return df[indicators]
    elif mode == "Single country" and country:
        return df[df["country"] == country][indicators]
    elif mode == "Cross-country average by year":
        return df.groupby("year")[indicators].mean().reset_index()[indicators]
    return df[indicators]


def render():
    st.header("Correlation & ML Analysis")

    # --- Load data ---
    datasets = list_saved_datasets()
    if not datasets and "current_dataset" not in st.session_state:
        st.info("No datasets found. Go to the **Download Data** page first.")
        return

    source = st.radio("Data source", ["Current session", "Saved dataset"], horizontal=True, key="corr_source")

    df = None
    if source == "Current session" and "current_dataset" in st.session_state:
        df = st.session_state["current_dataset"]
    elif source == "Saved dataset" and datasets:
        chosen = st.selectbox("Select dataset", datasets, key="corr_dataset")
        if chosen:
            df = load_dataset(chosen)
    else:
        st.warning("No data available.")
        return

    if df is None or df.empty:
        st.warning("Dataset is empty.")
        return

    indicators = _indicator_columns(df)
    if len(indicators) < 2:
        st.warning("Need at least 2 indicators for correlation analysis.")
        return

    # --- Data preparation ---
    st.subheader("\U00002699\uFE0F Data Preparation")
    countries = sorted(df["country"].unique()) if "country" in df.columns else []

    col1, col2 = st.columns(2)
    with col1:
        mode = st.selectbox(
            "Aggregation mode",
            ["All countries (pooled)", "Single country", "Cross-country average by year"],
        )
    with col2:
        selected_country = None
        if mode == "Single country" and countries:
            selected_country = st.selectbox("Country", countries)

    handle_missing = st.selectbox(
        "Handle missing values",
        ["Drop rows with any NaN", "Forward fill then drop", "Mean imputation"],
        key="corr_missing",
    )

    analysis_df = _prepare_analysis_data(df, mode, selected_country)

    if handle_missing == "Forward fill then drop":
        analysis_df = analysis_df.ffill().dropna()
    elif handle_missing == "Mean imputation":
        analysis_df = analysis_df.fillna(analysis_df.mean()).dropna()
    else:
        analysis_df = analysis_df.dropna()

    st.write(f"Analysis data: **{len(analysis_df)} rows** \u00d7 **{len(indicators)} indicators**")

    if len(analysis_df) < 5:
        st.error("Not enough data after filtering. Try a different aggregation mode or handle missing values differently.")
        return

    # --- Analysis method selection ---
    st.divider()
    st.subheader("\U0001F52C Analysis Methods")

    method = st.selectbox(
        "Choose analysis method",
        [
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
        ],
    )

    if st.button("Run Analysis", type="primary", use_container_width=True):
        _run_analysis(analysis_df, method, indicators, df, countries)


def _run_analysis(analysis_df, method, indicators, full_df, countries):
    """Execute the selected analysis method and display results."""

    if method == "Pearson Correlation":
        _run_correlation(analysis_df, "pearson")
    elif method == "Spearman Rank Correlation":
        _run_correlation(analysis_df, "spearman")
    elif method == "Kendall Tau Correlation":
        _run_correlation(analysis_df, "kendall")
    elif method == "Partial Correlation":
        _run_partial_correlation(analysis_df)
    elif method == "Mutual Information (non-linear)":
        _run_mutual_information(analysis_df)
    elif method.startswith("ML Feature Importance"):
        model_type = {
            "ML Feature Importance (Random Forest)": "random_forest",
            "ML Feature Importance (Gradient Boosting)": "gradient_boosting",
            "ML Feature Importance (Lasso)": "lasso",
            "ML Feature Importance (Elastic Net)": "elastic_net",
        }[method]
        _run_ml_importance(analysis_df, model_type)
    elif method == "PCA Analysis":
        _run_pca(analysis_df)
    elif method == "Deep Learning (Autoencoder)":
        _run_autoencoder(analysis_df)
    elif method == "Granger Causality Test":
        _run_granger(full_df, indicators, countries)
    elif method == "ML Cross-Validation Scores":
        _run_cv_scores(analysis_df, indicators)


def _render_correlation_heatmap(matrix: pd.DataFrame, title: str):
    """Render an interactive correlation heatmap."""
    labels = [_get_indicator_label(c) for c in matrix.columns]

    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=labels,
        y=labels,
        colorscale=DIVERGING_SCALE,
        zmid=0,
        text=np.round(matrix.values, 3),
        texttemplate="%{text}",
        textfont={"size": 10, "color": CREAM},
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        width=800,
        height=700,
        xaxis_tickangle=-45,
    )
    apply_steam_style(fig)
    st.plotly_chart(fig, use_container_width=True)


def _run_correlation(df, method):
    with st.spinner(f"Computing {method} correlation..."):
        corr_mat, pval_mat = correlation_with_pvalues(df, method)

    _render_correlation_heatmap(corr_mat, f"{method.title()} Correlation Matrix")

    # Top correlations
    st.subheader("Strongest Correlations")
    top = get_top_correlations(corr_mat, n=20)
    if not top.empty:
        top["indicator_1_name"] = top["indicator_1"].map(_get_indicator_label)
        top["indicator_2_name"] = top["indicator_2"].map(_get_indicator_label)
        st.dataframe(
            top[["indicator_1_name", "indicator_2_name", "correlation", "abs_correlation"]],
            use_container_width=True,
        )

    # P-values
    with st.expander("P-values (statistical significance)"):
        sig = pval_mat < 0.05
        st.write(f"**{sig.sum().sum() // 2}** pairs significant at p < 0.05")
        _render_correlation_heatmap(pval_mat, "P-values")


def _run_partial_correlation(df):
    with st.spinner("Computing partial correlations..."):
        pcorr = partial_correlation(df)

    if pcorr.empty:
        st.error("Could not compute partial correlations (need more rows than columns).")
        return

    _render_correlation_heatmap(pcorr, "Partial Correlation Matrix")

    st.subheader("Strongest Partial Correlations")
    top = get_top_correlations(pcorr, n=20)
    if not top.empty:
        top["indicator_1_name"] = top["indicator_1"].map(_get_indicator_label)
        top["indicator_2_name"] = top["indicator_2"].map(_get_indicator_label)
        st.dataframe(
            top[["indicator_1_name", "indicator_2_name", "correlation", "abs_correlation"]],
            use_container_width=True,
        )

    st.info("Partial correlations show the relationship between two variables after controlling for all other variables. This helps identify direct relationships.")


def _run_mutual_information(df):
    with st.spinner("Computing mutual information (this may take a moment)..."):
        mi_mat = mutual_information_matrix(df)

    _render_correlation_heatmap(mi_mat, "Mutual Information Matrix")

    st.subheader("Strongest Non-linear Dependencies")
    top = get_top_correlations(mi_mat, n=20)
    if not top.empty:
        top["indicator_1_name"] = top["indicator_1"].map(_get_indicator_label)
        top["indicator_2_name"] = top["indicator_2"].map(_get_indicator_label)
        top = top.rename(columns={"correlation": "mutual_info", "abs_correlation": "mutual_info_abs"})
        st.dataframe(
            top[["indicator_1_name", "indicator_2_name", "mutual_info"]],
            use_container_width=True,
        )

    st.info("Mutual information captures both linear and non-linear dependencies. Higher values indicate stronger dependency (regardless of direction).")


def _run_ml_importance(df, model_type):
    with st.spinner(f"Training {model_type} models for each indicator..."):
        importance_mat = ml_feature_importance_matrix(df, model_type)

    _render_correlation_heatmap(importance_mat, f"Feature Importance ({model_type})")

    st.subheader("Top Feature Importances")
    top = get_top_correlations(importance_mat, n=20)
    if not top.empty:
        top["target"] = top["indicator_1"].map(_get_indicator_label)
        top["feature"] = top["indicator_2"].map(_get_indicator_label)
        top = top.rename(columns={"correlation": "importance"})
        st.dataframe(
            top[["target", "feature", "importance"]],
            use_container_width=True,
        )

    st.info(f"Each cell (row i, col j) shows how important indicator j is for predicting indicator i using a {model_type.replace('_', ' ')} model.")


def _run_pca(df):
    with st.spinner("Running PCA..."):
        result = pca_analysis(df)

    if "error" in result:
        st.error(result["error"])
        return

    # Explained variance
    st.subheader("Explained Variance")
    var_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(result["n_components"])],
        "Explained Variance (%)": [v * 100 for v in result["explained_variance_ratio"]],
        "Cumulative (%)": [v * 100 for v in result["cumulative_variance"]],
    })

    fig = go.Figure()
    fig.add_bar(
        x=var_df["Component"],
        y=var_df["Explained Variance (%)"],
        name="Individual",
        marker_color=BRASS,
    )
    fig.add_scatter(
        x=var_df["Component"],
        y=var_df["Cumulative (%)"],
        name="Cumulative",
        mode="lines+markers",
        marker_color=EMBER,
        line_color=EMBER,
    )
    fig.update_layout(title="PCA Explained Variance", yaxis_title="%")
    apply_steam_style(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Loadings
    st.subheader("Component Loadings")
    loadings = result["loadings"]
    loadings.index = [_get_indicator_label(c) for c in loadings.index]

    n_show = min(5, loadings.shape[1])
    fig = px.imshow(
        loadings.iloc[:, :n_show],
        title="PCA Loadings (Top Components)",
        color_continuous_scale=DIVERGING_SCALE,
        aspect="auto",
    )
    apply_steam_style(fig)
    st.plotly_chart(fig, use_container_width=True)

    # 2D scatter
    if result["n_components"] >= 2:
        st.subheader("PCA 2D Projection")
        transformed = result["transformed"]
        fig = px.scatter(
            transformed,
            x="PC1",
            y="PC2",
            title="Data in PC1 vs PC2 Space",
            color_discrete_sequence=[BRASS],
        )
        apply_steam_style(fig)
        st.plotly_chart(fig, use_container_width=True)


def _run_autoencoder(df):
    st.write("Training an autoencoder to discover non-linear patterns in the data.")

    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.number_input("Epochs", min_value=50, max_value=1000, value=200, step=50)
    with col2:
        lr = st.number_input("Learning rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
    with col3:
        encoding_dim = st.number_input(
            "Encoding dimension",
            min_value=2,
            max_value=max(2, len(_indicator_columns(df))),
            value=max(2, len(_indicator_columns(df)) // 3),
        )

    progress_bar = st.progress(0)
    status = st.empty()

    def progress_callback(epoch, total, loss):
        progress_bar.progress(epoch / total)
        status.text(f"Epoch {epoch}/{total} - Loss: {loss:.6f}")

    with st.spinner("Training autoencoder..."):
        result = train_autoencoder(df, encoding_dim=encoding_dim, epochs=epochs, lr=lr, progress_callback=progress_callback)

    progress_bar.progress(1.0)
    status.text("Training complete!")

    if "error" in result:
        st.error(result["error"])
        return

    # Training loss
    st.subheader("Training Loss")
    fig = px.line(
        x=list(range(len(result["training_losses"]))),
        y=result["training_losses"],
        labels={"x": "Epoch", "y": "Loss"},
        title="Autoencoder Training Loss",
        color_discrete_sequence=[EMBER],
    )
    apply_steam_style(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Reconstruction errors
    st.subheader("Reconstruction Errors per Indicator")
    errors = result["reconstruction_errors"]
    errors.index = [_get_indicator_label(c) for c in errors.index]
    fig = px.bar(
        x=errors.index,
        y=errors.values,
        title="Per-Indicator Reconstruction Error",
        labels={"x": "Indicator", "y": "Mean Squared Error"},
        color_discrete_sequence=[BRASS],
    )
    fig.update_layout(xaxis_tickangle=-45)
    apply_steam_style(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.info("Higher reconstruction error = the indicator is harder to predict from other indicators = more independent/unique information.")

    # Importance matrix
    st.subheader("Non-linear Feature Importance (Perturbation-based)")
    importance = result["importance_matrix"]
    _render_correlation_heatmap(importance, "Autoencoder Perturbation Importance")

    top = get_top_correlations(importance, n=20)
    if not top.empty:
        top["target"] = top["indicator_1"].map(_get_indicator_label)
        top["feature"] = top["indicator_2"].map(_get_indicator_label)
        top = top.rename(columns={"correlation": "importance"})
        st.dataframe(
            top[["target", "feature", "importance"]],
            use_container_width=True,
        )


def _run_granger(full_df, indicators, countries):
    st.write("Test whether one indicator Granger-causes another (time-series causality).")

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Cause (X)", indicators, format_func=_get_indicator_label, key="granger_x")
    with col2:
        y_col = st.selectbox("Effect (Y)", indicators, format_func=_get_indicator_label, key="granger_y", index=min(1, len(indicators) - 1))

    country = st.selectbox("Country", countries, key="granger_country") if countries else None
    max_lag = st.slider("Max lag (years)", 1, 10, 3)

    if st.button("Run Granger Test"):
        test_df = full_df[full_df["country"] == country] if country else full_df
        test_df = test_df.sort_values("year")

        with st.spinner("Running Granger causality test..."):
            result = granger_causality_test(test_df, x_col, y_col, max_lag)

        if "error" in result:
            st.error(result["error"])
            return

        st.subheader(f"Results: {_get_indicator_label(x_col)} \u2192 {_get_indicator_label(y_col)}")

        if result["lag_results"]:
            rows = []
            for lag, vals in result["lag_results"].items():
                rows.append({
                    "Lag (years)": lag,
                    "F-statistic": round(vals["f_statistic"], 4),
                    "P-value": round(vals["p_value"], 6),
                    "Significant (5%)": "Yes" if vals["significant_5pct"] else "No",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            any_sig = any(v["significant_5pct"] for v in result["lag_results"].values())
            if any_sig:
                st.success(f"Evidence that **{_get_indicator_label(x_col)}** Granger-causes **{_get_indicator_label(y_col)}**")
            else:
                st.warning(f"No significant evidence of Granger causality at 5% level.")
        else:
            st.warning("Could not compute results. Try with more data or fewer lags.")


def _run_cv_scores(df, indicators):
    st.write("Evaluate how well other indicators predict a target using cross-validation.")

    target = st.selectbox("Target indicator", indicators, format_func=_get_indicator_label, key="cv_target")
    model_type = st.selectbox(
        "Model",
        ["random_forest", "gradient_boosting", "lasso", "elastic_net"],
        format_func=lambda x: x.replace("_", " ").title(),
        key="cv_model",
    )
    cv_folds = st.slider("CV folds", 2, 10, 5, key="cv_folds")

    if st.button("Run Cross-Validation"):
        with st.spinner("Running cross-validation..."):
            result = ml_cross_validated_scores(df, target, model_type, cv_folds)

        if "error" in result:
            st.error(result["error"])
            return

        st.subheader("Cross-Validation Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean R\u00b2", f"{result['r2_mean']:.4f}")
        col2.metric("Std R\u00b2", f"\u00b1{result['r2_std']:.4f}")
        col3.metric("Samples", result['n_samples'])

        fig = px.bar(
            x=[f"Fold {i+1}" for i in range(len(result['r2_scores']))],
            y=result['r2_scores'],
            title=f"R\u00b2 Scores per Fold ({model_type.replace('_', ' ').title()})",
            labels={"x": "Fold", "y": "R\u00b2 Score"},
            color_discrete_sequence=[BRASS],
        )
        fig.add_hline(y=result['r2_mean'], line_dash="dash", line_color=EMBER,
                       annotation_text=f"Mean: {result['r2_mean']:.4f}",
                       annotation_font_color=CREAM)
        apply_steam_style(fig)
        st.plotly_chart(fig, use_container_width=True)

        if result['r2_mean'] > 0.7:
            st.success(f"Good predictability (R\u00b2 = {result['r2_mean']:.3f}). Other indicators strongly predict **{_get_indicator_label(target)}**.")
        elif result['r2_mean'] > 0.3:
            st.info(f"Moderate predictability (R\u00b2 = {result['r2_mean']:.3f}).")
        else:
            st.warning(f"Low predictability (R\u00b2 = {result['r2_mean']:.3f}). This indicator may be largely independent of the others.")
