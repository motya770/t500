"""Inflation-Stock Correlation Models page.

Lets the user pick an inflation indicator and a stock ticker (or any
economic target), choose from PyTorch deep-learning or scikit-learn
boosting models, tune hyperparameters, and visualise predictions,
metrics, feature importances, and regime probabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data_sources.world_bank import (
    list_saved_datasets,
    load_dataset,
    get_all_indicators,
    INDICATOR_CATEGORIES,
)
from data_sources.stock_data import (
    download_stock_data,
    merge_stock_with_economic,
    STOCK_PRESETS,
)
from analysis.inflation_stock_models import (
    run_pytorch_model,
    run_boosting_model,
)


# -- helpers -----------------------------------------------------------------

def _get_indicator_label(code: str) -> str:
    names = st.session_state.get("indicator_names", {})
    if code in names:
        return names[code]
    return get_all_indicators().get(code, code)


def _indicator_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("country", "year")]


# -- page entry point --------------------------------------------------------

def render():
    st.header("Inflation-Stock Correlation Models")
    st.markdown(
        "Investigate the relationship between **inflation** (or any economic "
        "indicator) and **stock prices** using deep-learning and boosting models."
    )

    # ================================================================
    # 1. DATA SETUP
    # ================================================================
    st.subheader("1 - Data Setup")

    # -- Economic data ------------------------------------------------
    datasets = list_saved_datasets()
    has_session = "current_dataset" in st.session_state

    if not datasets and not has_session:
        st.info("No datasets found. Go to **Download Data** first to fetch economic indicators.")
        return

    econ_source = st.radio(
        "Economic data source",
        ["Current session", "Saved dataset"],
        horizontal=True,
        key="is_econ_source",
    )

    econ_df = None
    if econ_source == "Current session" and has_session:
        econ_df = st.session_state["current_dataset"]
    elif econ_source == "Saved dataset" and datasets:
        chosen = st.selectbox("Select dataset", datasets, key="is_econ_ds")
        if chosen:
            econ_df = load_dataset(chosen)

    if econ_df is None or econ_df.empty:
        st.warning("No economic data available.")
        return

    indicators = _indicator_columns(econ_df)
    if not indicators:
        st.warning("No indicator columns found in the dataset.")
        return

    # -- Country filter -----------------------------------------------
    countries = sorted(econ_df["country"].unique()) if "country" in econ_df.columns else []
    if countries:
        selected_country = st.selectbox(
            "Select country (time-series models need single-country data)",
            countries,
            key="is_country",
        )
        work_df = econ_df[econ_df["country"] == selected_country].sort_values("year").reset_index(drop=True)
    else:
        work_df = econ_df.sort_values("year").reset_index(drop=True) if "year" in econ_df.columns else econ_df

    # -- Stock data ---------------------------------------------------
    st.markdown("---")
    use_stock = st.checkbox("Download stock / ETF data as target variable", value=True, key="is_use_stock")

    stock_col_name = None
    if use_stock:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            ticker = st.selectbox(
                "Ticker",
                list(STOCK_PRESETS.keys()),
                format_func=lambda t: f"{t} — {STOCK_PRESETS[t]}",
                key="is_ticker",
            )
        with col_t2:
            stock_metric = st.selectbox(
                "Stock metric to predict",
                ["annual_return_pct", "avg_price", "end_price", "volatility"],
                format_func=lambda m: m.replace("_", " ").title(),
                key="is_stock_metric",
            )

        if st.button("Download stock data", key="is_dl_stock"):
            year_min = int(work_df["year"].min()) if "year" in work_df.columns else 2000
            year_max = int(work_df["year"].max()) if "year" in work_df.columns else 2024
            with st.spinner(f"Downloading {ticker} data …"):
                try:
                    stock_df = download_stock_data(ticker, year_min, year_max)
                    if stock_df.empty:
                        st.error("No stock data returned. Check the ticker symbol.")
                        return
                    st.session_state["is_stock_df"] = stock_df
                    st.session_state["is_stock_ticker"] = ticker
                    st.success(f"Downloaded {len(stock_df)} years of {ticker} data.")
                except Exception as exc:
                    st.error(f"Failed to download stock data: {exc}")
                    return

        if "is_stock_df" in st.session_state:
            stock_df = st.session_state["is_stock_df"]
            stock_col_name = f"{st.session_state.get('is_stock_ticker', 'stock')}_{stock_metric}"
            work_df = merge_stock_with_economic(
                stock_df, work_df,
                stock_col=stock_metric,
                stock_col_name=stock_col_name,
            )
            st.write(f"Merged data: **{len(work_df)} rows**")
            with st.expander("Preview merged data"):
                st.dataframe(work_df.head(20), use_container_width=True)

    # -- Feature / target selection -----------------------------------
    st.markdown("---")
    available_cols = _indicator_columns(work_df)
    if len(available_cols) < 2:
        st.warning("Need at least 2 columns (1 feature + 1 target) in the merged data.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        default_target = stock_col_name if stock_col_name and stock_col_name in available_cols else available_cols[-1]
        target_col = st.selectbox(
            "Target variable (Y)",
            available_cols,
            index=available_cols.index(default_target) if default_target in available_cols else 0,
            format_func=lambda c: _get_indicator_label(c) if c in get_all_indicators() else c,
            key="is_target",
        )
    with col_b:
        remaining = [c for c in available_cols if c != target_col]
        feature_cols = st.multiselect(
            "Feature variables (X)",
            remaining,
            default=remaining,
            format_func=lambda c: _get_indicator_label(c) if c in get_all_indicators() else c,
            key="is_features",
        )

    if not feature_cols:
        st.warning("Select at least one feature variable.")
        return

    # Handle missing values
    missing_mode = st.selectbox(
        "Handle missing values",
        ["Drop rows with any NaN", "Forward fill then drop", "Mean imputation"],
        key="is_missing",
    )
    analysis_df = work_df[["year"] + feature_cols + [target_col]].copy() if "year" in work_df.columns else work_df[feature_cols + [target_col]].copy()
    if missing_mode == "Forward fill then drop":
        analysis_df = analysis_df.ffill().dropna()
    elif missing_mode == "Mean imputation":
        analysis_df = analysis_df.fillna(analysis_df.mean()).dropna()
    else:
        analysis_df = analysis_df.dropna()

    st.write(
        f"Analysis data: **{len(analysis_df)} rows** / "
        f"**{len(feature_cols)} features** / target = "
        f"*{_get_indicator_label(target_col) if target_col in get_all_indicators() else target_col}*"
    )

    if len(analysis_df) < 6:
        st.error("Not enough data after cleaning (need at least 6 rows). Try different indicators, country, or missing-value strategy.")
        return

    # ================================================================
    # 2. MODEL SELECTION
    # ================================================================
    st.divider()
    st.subheader("2 - Model Selection")

    model_category = st.radio(
        "Model family",
        ["PyTorch Deep Learning", "Scikit-learn Boosting"],
        horizontal=True,
        key="is_model_cat",
    )

    if model_category == "PyTorch Deep Learning":
        model_type = st.selectbox(
            "Model",
            ["lstm", "gru", "transformer", "tcn", "regime_switching"],
            format_func={
                "lstm": "LSTM (Long Short-Term Memory)",
                "gru": "GRU (Gated Recurrent Unit)",
                "transformer": "Transformer Encoder",
                "tcn": "Temporal Convolutional Network",
                "regime_switching": "Neural Regime Switching",
            }.get,
            key="is_pt_model",
        )

        st.markdown("**Hyperparameters**")
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            seq_length = st.number_input("Sequence length", 2, 10, 3, key="is_seq")
        with h2:
            epochs = st.number_input("Epochs", 50, 1000, 300, step=50, key="is_epochs")
        with h3:
            lr = st.number_input("Learning rate", 0.0001, 0.01, 0.001, format="%.4f", key="is_lr")
        with h4:
            hidden_dim = st.number_input("Hidden dim", 16, 256, 64, step=16, key="is_hdim")

        extra_kwargs: dict = {}
        if model_type == "regime_switching":
            extra_kwargs["n_regimes"] = st.slider("Number of regimes", 2, 5, 3, key="is_regimes")

        if st.button("Train Model", type="primary", use_container_width=True, key="is_run_pt"):
            _run_pytorch(
                analysis_df, target_col, feature_cols, model_type,
                seq_length, epochs, lr, hidden_dim, **extra_kwargs,
            )

    else:  # Boosting
        model_type = st.selectbox(
            "Model",
            ["gradient_boosting", "adaboost", "hist_gradient_boosting"],
            format_func={
                "gradient_boosting": "Gradient Boosting Regressor",
                "adaboost": "AdaBoost Regressor",
                "hist_gradient_boosting": "Hist Gradient Boosting (LightGBM-style)",
            }.get,
            key="is_boost_model",
        )

        st.markdown("**Hyperparameters**")
        b1, b2, b3 = st.columns(3)
        with b1:
            n_lags = st.number_input("Number of lags", 1, 10, 3, key="is_nlags")
        with b2:
            n_estimators = st.number_input("Estimators", 50, 500, 100, step=50, key="is_nest")
        with b3:
            cv_folds = st.number_input("CV folds", 2, 10, 5, key="is_cvf")

        if st.button("Train Model", type="primary", use_container_width=True, key="is_run_boost"):
            _run_boosting(
                analysis_df, target_col, feature_cols, model_type,
                n_lags, n_estimators, cv_folds,
            )


# ================================================================
# Rendering helpers
# ================================================================


def _run_pytorch(df, target_col, feature_cols, model_type,
                 seq_length, epochs, lr, hidden_dim, **kwargs):
    """Train a PyTorch model and render results."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def _progress(epoch, total, t_loss, v_loss):
        progress_bar.progress(min(epoch / total, 1.0))
        status_text.text(f"Epoch {epoch}/{total}  —  train loss {t_loss:.6f}  val loss {v_loss:.6f}")

    model_labels = {
        "lstm": "LSTM",
        "gru": "GRU",
        "transformer": "Transformer",
        "tcn": "TCN",
        "regime_switching": "Regime Switching",
    }

    with st.spinner(f"Training {model_labels.get(model_type, model_type)} model …"):
        result = run_pytorch_model(
            df, target_col, feature_cols,
            model_type=model_type,
            seq_length=seq_length,
            epochs=epochs,
            lr=lr,
            hidden_dim=hidden_dim,
            progress_callback=_progress,
            **kwargs,
        )

    progress_bar.progress(1.0)
    status_text.text("Training complete.")

    if "error" in result:
        st.error(result["error"])
        return

    _render_results(result, target_col, feature_cols, is_pytorch=True)


def _run_boosting(df, target_col, feature_cols, model_type,
                  n_lags, n_estimators, cv_folds):
    """Train a boosting model and render results."""
    model_labels = {
        "gradient_boosting": "Gradient Boosting",
        "adaboost": "AdaBoost",
        "hist_gradient_boosting": "Hist Gradient Boosting",
    }
    with st.spinner(f"Training {model_labels.get(model_type, model_type)} …"):
        result = run_boosting_model(
            df, target_col, feature_cols,
            model_type=model_type,
            n_lags=n_lags,
            n_estimators=n_estimators,
            cv_folds=cv_folds,
        )

    if "error" in result:
        st.error(result["error"])
        return

    _render_results(result, target_col, feature_cols, is_pytorch=False)


# ------------------------------------------------------------------ results

def _render_results(result: dict, target_col: str, feature_cols: list[str], is_pytorch: bool):
    """Unified rendering for both PyTorch and boosting results."""
    st.divider()
    st.subheader("3 - Results")

    target_label = _get_indicator_label(target_col) if target_col in get_all_indicators() else target_col

    # ---- Metrics cards ----
    metrics = result["metrics"]
    cols = st.columns(3)
    if is_pytorch:
        cols[0].metric("Train R²", f"{metrics.get('train_r2', 0):.4f}")
        cols[1].metric("Val R²", f"{metrics.get('val_r2', 0):.4f}" if "val_r2" in metrics else "N/A")
        cols[2].metric("Val MAE", f"{metrics.get('val_mae', 0):.4f}" if "val_mae" in metrics else "N/A")
    else:
        cols[0].metric("R² (full)", f"{metrics.get('r2', 0):.4f}")
        cols[1].metric("MSE", f"{metrics.get('mse', 0):.4f}")
        cols[2].metric("MAE", f"{metrics.get('mae', 0):.4f}")

    # ---- Predictions vs Actual chart ----
    preds = result["predictions"]
    pred_df = pd.DataFrame({
        "Year": preds["years"],
        "Actual": preds["actual"],
        "Predicted": preds["predicted"],
    })

    fig = go.Figure()
    fig.add_scatter(x=pred_df["Year"], y=pred_df["Actual"], mode="lines+markers", name="Actual")
    fig.add_scatter(x=pred_df["Year"], y=pred_df["Predicted"], mode="lines+markers", name="Predicted")

    if is_pytorch and "train_size" in preds:
        split_year = pred_df["Year"].iloc[preds["train_size"]] if preds["train_size"] < len(pred_df) else None
        if split_year is not None:
            fig.add_vline(x=split_year, line_dash="dash", line_color="gray",
                          annotation_text="Train / Val split")

    fig.update_layout(
        title=f"Predicted vs Actual — {target_label}",
        xaxis_title="Year",
        yaxis_title=target_label,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Training history (PyTorch only) ----
    if is_pytorch and "training_history" in result:
        with st.expander("Training history"):
            hist = result["training_history"]
            hist_df = pd.DataFrame({
                "Epoch": list(range(len(hist["train_loss"]))),
                "Train Loss": hist["train_loss"],
                "Val Loss": hist["val_loss"],
            })
            fig_loss = go.Figure()
            fig_loss.add_scatter(x=hist_df["Epoch"], y=hist_df["Train Loss"], name="Train")
            fig_loss.add_scatter(x=hist_df["Epoch"], y=hist_df["Val Loss"], name="Validation")
            fig_loss.update_layout(title="Loss Curve", yaxis_title="MSE Loss")
            st.plotly_chart(fig_loss, use_container_width=True)

    # ---- Cross-validation results (boosting only) ----
    if not is_pytorch and "cv_results" in result:
        with st.expander("Cross-validation folds"):
            cv = result["cv_results"]
            cv_df = pd.DataFrame(cv)
            fig_cv = px.bar(cv_df, x="fold", y="r2", title="R² per Fold (Time-Series CV)")
            mean_r2 = cv_df["r2"].mean()
            fig_cv.add_hline(y=mean_r2, line_dash="dash",
                             annotation_text=f"Mean R²: {mean_r2:.4f}")
            st.plotly_chart(fig_cv, use_container_width=True)
            st.dataframe(cv_df, use_container_width=True)

    # ---- Feature importances (boosting only) ----
    if not is_pytorch and "feature_importances" in result:
        with st.expander("Feature importances"):
            imp = pd.DataFrame(result["feature_importances"])
            if not imp.empty:
                # Shorten labels
                imp["label"] = imp["feature"].apply(
                    lambda f: _get_indicator_label(f.split("_lag")[0]) + (
                        f" (lag {f.split('_lag')[1]})" if "_lag" in f else ""
                    ) if any(f.startswith(c) for c in get_all_indicators()) else f
                )
                fig_imp = px.bar(
                    imp.head(15),
                    x="importance", y="label", orientation="h",
                    title="Top Feature Importances",
                )
                fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp, use_container_width=True)

    # ---- Regime probabilities (regime switching only) ----
    if "regime_probabilities" in result:
        with st.expander("Regime probabilities over time"):
            regimes = np.array(result["regime_probabilities"])
            years = preds["years"]
            n_regimes = regimes.shape[1]
            fig_reg = go.Figure()
            for r in range(n_regimes):
                fig_reg.add_scatter(
                    x=years, y=regimes[:, r],
                    mode="lines+markers",
                    name=f"Regime {r + 1}",
                    stackgroup="one",
                )
            fig_reg.update_layout(
                title="Regime Probabilities Over Time",
                xaxis_title="Year",
                yaxis_title="Probability",
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_reg, use_container_width=True)

            dominant = np.argmax(regimes, axis=1) + 1
            st.write("**Dominant regime per year:**")
            regime_df = pd.DataFrame({"Year": years, "Dominant Regime": dominant})
            st.dataframe(regime_df, use_container_width=True)

    # ---- Scatter: Actual vs Predicted ----
    with st.expander("Actual vs Predicted scatter"):
        scatter_df = pd.DataFrame({
            "Actual": preds["actual"],
            "Predicted": preds["predicted"],
        })
        fig_sc = px.scatter(
            scatter_df, x="Actual", y="Predicted",
            title="Actual vs Predicted",
            trendline="ols",
        )
        min_val = min(scatter_df.min())
        max_val = max(scatter_df.max())
        fig_sc.add_scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Perfect prediction",
            line=dict(dash="dash", color="gray"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)
