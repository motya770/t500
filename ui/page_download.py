"""Data Download page for the Streamlit app."""

import streamlit as st
import pandas as pd
from data_sources.world_bank import (
    INDICATOR_CATEGORIES,
    get_country_groups,
    download_multiple_indicators,
    save_dataset,
    get_all_indicators,
)


def render():
    st.header("\U0001F682 Download Economic Data")

    data_source = st.radio(
        "Data Source",
        ["World Bank API", "HuggingFace Dataset"],
        horizontal=True,
        help=(
            "World Bank API fetches live data (may be slow). "
            "HuggingFace uses a cached offline dataset with 1,400+ indicators."
        ),
    )

    if data_source == "World Bank API":
        _render_world_bank_download()
    else:
        _render_huggingface_download()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _render_indicator_selection(key_prefix=""):
    """Render indicator category checkboxes. Returns list of selected codes."""
    all_indicators = get_all_indicators()
    selected_codes = []

    select_all_global = st.checkbox(
        f"Select all indicators ({len(all_indicators)} total)",
        key=f"{key_prefix}select_all_global",
    )

    cols = st.columns(2)
    for idx, (category, indicators) in enumerate(INDICATOR_CATEGORIES.items()):
        col = cols[idx % 2]
        with col:
            with st.expander(f"{category} ({len(indicators)} indicators)"):
                select_all = st.checkbox(
                    f"Select all {category}",
                    value=select_all_global,
                    key=f"{key_prefix}cat_{category}",
                )
                for code, name in indicators.items():
                    checked = st.checkbox(
                        f"{name}",
                        value=select_all or select_all_global,
                        key=f"{key_prefix}ind_{code}",
                        help=f"Code: {code}",
                    )
                    if checked:
                        selected_codes.append(code)

    return selected_codes


def _render_country_selection(key_prefix=""):
    """Render country group / manual entry selection. Returns list of ISO3 codes."""
    country_groups = get_country_groups()
    selection_mode = st.radio(
        "Selection mode",
        ["Predefined groups", "Manual entry"],
        horizontal=True,
        key=f"{key_prefix}country_mode",
    )

    selected_countries = []
    if selection_mode == "Predefined groups":
        chosen_groups = st.multiselect(
            "Choose country groups",
            options=list(country_groups.keys()),
            default=["G7"],
            key=f"{key_prefix}country_groups",
        )
        for group in chosen_groups:
            selected_countries.extend(country_groups[group])
        selected_countries = list(set(selected_countries))

        if selected_countries:
            st.write(f"Countries: {', '.join(sorted(selected_countries))}")
    else:
        manual_input = st.text_area(
            "Enter ISO3 country codes (comma-separated)",
            value="USA, GBR, DEU, FRA, JPN, CHN, IND, BRA",
            help="Use ISO 3166-1 alpha-3 codes (e.g., USA, GBR, DEU)",
            key=f"{key_prefix}manual_countries",
        )
        selected_countries = [c.strip() for c in manual_input.split(",") if c.strip()]

    return selected_countries


def _show_download_results(df, dataset_name, failed_indicators, indicator_names):
    """Display download results: warnings, preview, summary, session state."""
    if failed_indicators:
        st.warning(
            f"**{len(failed_indicators)} indicator(s) failed to download** "
            f"(not available) and were skipped:\n"
            + "\n".join(f"- {name}" for name in failed_indicators)
        )

    if df.empty:
        st.error("No data returned. Try different indicators or countries.")
        return

    path = save_dataset(df, dataset_name)
    st.success(f"Dataset saved to `{path}` ({len(df)} rows, {len(df.columns)} columns)")

    st.subheader("Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    with st.expander("Dataset Summary"):
        st.write(f"**Rows:** {len(df)}")
        st.write(f"**Columns:** {len(df.columns)}")
        st.write(f"**Countries:** {df['country'].nunique()}")
        st.write(f"**Year range:** {df['year'].min()} - {df['year'].max()}")
        st.write("**Missing values per column:**")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.dataframe(missing.to_frame("missing_count"))
        else:
            st.write("No missing values!")

    st.session_state["current_dataset"] = df
    st.session_state["current_dataset_name"] = dataset_name
    st.session_state["indicator_names"] = indicator_names


# ---------------------------------------------------------------------------
# World Bank API download
# ---------------------------------------------------------------------------

def _render_world_bank_download():
    st.write("Select indicators and countries to download from the World Bank Open Data API.")

    st.subheader("\U0001F527 1. Select Indicators")
    all_indicators = get_all_indicators()
    selected_codes = _render_indicator_selection(key_prefix="wb_")

    if selected_codes:
        st.success(f"Selected {len(selected_codes)} indicators")
    else:
        st.info("Select at least one indicator to proceed.")

    st.subheader("\U0001F30D 2. Select Countries")
    selected_countries = _render_country_selection(key_prefix="wb_")

    st.subheader("\U0001F4C5 3. Select Year Range")
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start year", min_value=1960, max_value=2025, value=2000, key="wb_start")
    with col2:
        end_year = st.number_input("End year", min_value=1960, max_value=2025, value=2025, key="wb_end")

    if start_year > end_year:
        st.error("Start year must be before end year.")
        return

    st.subheader("\U0001F3F7\uFE0F 4. Name Your Dataset")
    dataset_name = st.text_input(
        "Dataset name", value="economic_data",
        help="This will be used as the filename for saving.",
        key="wb_dataset_name",
    )

    st.divider()

    if not selected_codes:
        st.warning("Please select at least one indicator.")
        return
    if not selected_countries:
        st.warning("Please select at least one country.")
        return

    st.write(f"**Summary:** {len(selected_codes)} indicators \u00d7 {len(selected_countries)} countries \u00d7 {end_year - start_year + 1} years")

    if st.button("\U0001F682 Download Data", type="primary", use_container_width=True, key="wb_download"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current, total, label):
            progress_bar.progress(current / total)
            status_text.text(f"Downloading: {label} ({current + 1}/{total})")

        try:
            df, failed_indicators = download_multiple_indicators(
                indicator_codes=selected_codes,
                countries=selected_countries,
                start_year=start_year,
                end_year=end_year,
                progress_callback=progress_callback,
            )

            progress_bar.progress(1.0)
            status_text.text("Download complete!")

            indicator_names = {
                code: all_indicators.get(code, code) for code in selected_codes
            }
            _show_download_results(df, dataset_name, failed_indicators, indicator_names)

        except Exception as e:
            st.error(f"Download failed: {e}")


# ---------------------------------------------------------------------------
# HuggingFace dataset download
# ---------------------------------------------------------------------------

def _render_huggingface_download():
    from data_sources.huggingface_data import (
        is_cache_available,
        get_cache_info,
        download_hf_dataset,
        get_hf_indicator_catalog,
        download_from_hf,
    )

    st.write(
        "Use the HuggingFace World Development Indicators dataset "
        "(1,400+ indicators, offline after first download)."
    )

    # --- Cache management ---
    st.subheader("\U0001F4BE 0. Dataset Cache")

    if is_cache_available():
        info = get_cache_info()
        st.success(
            f"Cache available: {info['size_mb']} MB, "
            f"{info.get('indicator_count', '?')} indicators"
        )
        if st.button("Re-download (update cache)", key="hf_redownload"):
            with st.spinner("Downloading dataset from HuggingFace..."):
                download_hf_dataset(force=True)
            st.rerun()
    else:
        st.info("The HuggingFace dataset needs to be downloaded once (~64 MB).")
        if st.button("Download Dataset Cache", type="primary", key="hf_cache_download"):
            with st.spinner("Downloading dataset from HuggingFace (~64 MB)..."):
                try:
                    download_hf_dataset()
                    st.rerun()
                except Exception as e:
                    st.error(f"Download failed: {e}")
        return  # can't proceed without cache

    # --- Indicator selection ---
    st.subheader("\U0001F527 1. Select Indicators")

    indicator_mode = st.radio(
        "Indicator selection mode",
        ["Curated indicators", "Browse all HF indicators"],
        horizontal=True,
        key="hf_ind_mode",
    )

    all_wb_indicators = get_all_indicators()
    selected_codes = []

    if indicator_mode == "Curated indicators":
        selected_codes = _render_indicator_selection(key_prefix="hf_")
    else:
        hf_catalog = get_hf_indicator_catalog()
        search_term = st.text_input(
            "Search indicators",
            placeholder="e.g., GDP, education, mortality",
            key="hf_search",
        )
        if search_term:
            term = search_term.lower()
            matches = {
                k: v for k, v in hf_catalog.items()
                if term in v.lower() or term in k.lower()
            }
        else:
            matches = {}

        if matches:
            selected_codes = st.multiselect(
                f"Select from {len(matches)} matching indicators",
                options=list(matches.keys()),
                format_func=lambda x: f"{matches[x]} ({x})",
                key="hf_multiselect",
            )
        elif search_term:
            st.info("No indicators match your search.")

    if selected_codes:
        st.success(f"Selected {len(selected_codes)} indicators")
    else:
        st.info("Select at least one indicator to proceed.")

    # --- Country selection ---
    st.subheader("\U0001F30D 2. Select Countries")
    selected_countries = _render_country_selection(key_prefix="hf_")

    # --- Year range ---
    st.subheader("\U0001F4C5 3. Select Year Range")
    st.caption("HuggingFace dataset contains data from 1960 to ~2020.")
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start year", min_value=1960, max_value=2025, value=2000, key="hf_start")
    with col2:
        end_year = st.number_input("End year", min_value=1960, max_value=2025, value=2020, key="hf_end")

    if start_year > end_year:
        st.error("Start year must be before end year.")
        return

    # --- Dataset name ---
    st.subheader("\U0001F3F7\uFE0F 4. Name Your Dataset")
    dataset_name = st.text_input(
        "Dataset name", value="hf_economic_data",
        help="This will be used as the filename for saving.",
        key="hf_dataset_name",
    )

    st.divider()

    if not selected_codes:
        st.warning("Please select at least one indicator.")
        return
    if not selected_countries:
        st.warning("Please select at least one country.")
        return

    st.write(f"**Summary:** {len(selected_codes)} indicators \u00d7 {len(selected_countries)} countries \u00d7 {end_year - start_year + 1} years")

    if st.button("\U0001F682 Load Data", type="primary", use_container_width=True, key="hf_download"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current, total, label):
            progress_bar.progress(current / total if total else 1.0)
            status_text.text(label)

        try:
            df, failed_indicators = download_from_hf(
                indicator_codes=selected_codes,
                countries=selected_countries,
                start_year=start_year,
                end_year=end_year,
                progress_callback=progress_callback,
            )

            progress_bar.progress(1.0)
            status_text.text("Done!")

            hf_catalog = get_hf_indicator_catalog()
            indicator_names = {
                code: hf_catalog.get(code, all_wb_indicators.get(code, code))
                for code in selected_codes
            }
            _show_download_results(df, dataset_name, failed_indicators, indicator_names)

        except Exception as e:
            st.error(f"Failed: {e}")
