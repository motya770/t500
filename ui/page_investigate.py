"""Dataset Investigation page — inspect structure and browse data with paging."""

import streamlit as st
import pandas as pd
from data_sources.world_bank import list_saved_datasets, load_dataset, get_all_indicators


def _get_indicator_label(code: str) -> str:
    names = st.session_state.get("indicator_names", {})
    if code in names:
        return names[code]
    all_ind = get_all_indicators()
    return all_ind.get(code, code)


PAGE_SIZE = 20


def render():
    st.header("Investigate Datasets")

    datasets = list_saved_datasets()
    if not datasets and "current_dataset" not in st.session_state:
        st.info("No datasets found. Go to a **Download Data** page first.")
        return

    source = st.radio("Data source", ["Saved dataset", "Current session"], horizontal=True, key="inv_source")

    df = None
    dataset_label = ""
    if source == "Current session" and "current_dataset" in st.session_state:
        df = st.session_state["current_dataset"]
        dataset_label = st.session_state.get("current_dataset_name", "session data")
    elif source == "Saved dataset" and datasets:
        chosen = st.selectbox("Select dataset", datasets, key="inv_dataset")
        if chosen:
            df = load_dataset(chosen)
            dataset_label = chosen
    else:
        st.warning("No data available. Download data first.")
        return

    if df is None or df.empty:
        st.warning("Dataset is empty.")
        return

    st.subheader(f"Dataset: {dataset_label}")

    # --- Structure overview ---
    st.markdown("#### Structure")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    # Column info table
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(dt) for dt in df.dtypes],
        "Non-Null": [int(df[c].notna().sum()) for c in df.columns],
        "Null": [int(df[c].isna().sum()) for c in df.columns],
        "Null %": [f"{df[c].isna().mean() * 100:.1f}%" for c in df.columns],
        "Unique": [int(df[c].nunique()) for c in df.columns],
    })

    # Add human-readable label for indicator columns
    labels = []
    for c in df.columns:
        if c not in ("country", "year"):
            label = _get_indicator_label(c)
            labels.append(label if label != c else "")
        else:
            labels.append("")
    col_info["Label"] = labels

    st.dataframe(col_info, use_container_width=True, hide_index=True)

    # Numeric summary
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        with st.expander("Numeric Summary (describe)", expanded=False):
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

    # --- Paginated data table ---
    st.markdown("#### Data")

    # Optional column filter
    all_cols = df.columns.tolist()
    selected_cols = st.multiselect(
        "Columns to display",
        all_cols,
        default=all_cols,
        key="inv_cols",
    )
    if not selected_cols:
        selected_cols = all_cols

    view_df = df[selected_cols]
    total_rows = len(view_df)
    total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)

    col_left, col_mid, col_right = st.columns([1, 2, 1])
    with col_mid:
        page = st.number_input(
            f"Page (1–{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key="inv_page",
        )

    start = (page - 1) * PAGE_SIZE
    end = min(start + PAGE_SIZE, total_rows)

    st.caption(f"Showing rows {start + 1}–{end} of {total_rows:,}")
    st.dataframe(view_df.iloc[start:end], use_container_width=True, hide_index=False)
