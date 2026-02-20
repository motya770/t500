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
    st.write("Select indicators and countries to download from the World Bank Open Data API.")

    # --- Indicator selection ---
    st.subheader("\U0001F527 1. Select Indicators")

    all_indicators = get_all_indicators()
    selected_codes = []

    # Global select all
    select_all_global = st.checkbox(
        f"Select all indicators ({len(all_indicators)} total)",
        key="select_all_global",
    )

    # Quick select by category
    cols = st.columns(2)
    for idx, (category, indicators) in enumerate(INDICATOR_CATEGORIES.items()):
        col = cols[idx % 2]
        with col:
            with st.expander(f"{category} ({len(indicators)} indicators)"):
                select_all = st.checkbox(
                    f"Select all {category}",
                    value=select_all_global,
                    key=f"cat_{category}",
                )
                for code, name in indicators.items():
                    checked = st.checkbox(
                        f"{name}",
                        value=select_all or select_all_global,
                        key=f"ind_{code}",
                        help=f"Code: {code}",
                    )
                    if checked:
                        selected_codes.append(code)

    if selected_codes:
        st.success(f"Selected {len(selected_codes)} indicators")
    else:
        st.info("Select at least one indicator to proceed.")

    # --- Country selection ---
    st.subheader("\U0001F30D 2. Select Countries")

    country_groups = get_country_groups()
    selection_mode = st.radio(
        "Selection mode",
        ["Predefined groups", "Manual entry"],
        horizontal=True,
    )

    selected_countries = []
    if selection_mode == "Predefined groups":
        chosen_groups = st.multiselect(
            "Choose country groups",
            options=list(country_groups.keys()),
            default=["G7"],
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
        )
        selected_countries = [c.strip() for c in manual_input.split(",") if c.strip()]

    # --- Year range ---
    st.subheader("\U0001F4C5 3. Select Year Range")
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start year", min_value=1960, max_value=2023, value=2000)
    with col2:
        end_year = st.number_input("End year", min_value=1960, max_value=2023, value=2023)

    if start_year > end_year:
        st.error("Start year must be before end year.")
        return

    # --- Dataset name ---
    st.subheader("\U0001F3F7\uFE0F 4. Name Your Dataset")
    dataset_name = st.text_input(
        "Dataset name",
        value="economic_data",
        help="This will be used as the filename for saving.",
    )

    # --- Download button ---
    st.divider()

    if not selected_codes:
        st.warning("Please select at least one indicator.")
        return
    if not selected_countries:
        st.warning("Please select at least one country.")
        return

    st.write(f"**Summary:** {len(selected_codes)} indicators × {len(selected_countries)} countries × {end_year - start_year + 1} years")

    if st.button("\U0001F682 Download Data", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current, total, label):
            progress_bar.progress(current / total)
            status_text.text(f"Downloading: {label} ({current + 1}/{total})")

        try:
            df = download_multiple_indicators(
                indicator_codes=selected_codes,
                countries=selected_countries,
                start_year=start_year,
                end_year=end_year,
                progress_callback=progress_callback,
            )

            progress_bar.progress(1.0)
            status_text.text("Download complete!")

            if df.empty:
                st.error("No data returned. Try different indicators or countries.")
                return

            # Save dataset
            path = save_dataset(df, dataset_name)
            st.success(f"Dataset saved to `{path}` ({len(df)} rows, {len(df.columns)} columns)")

            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head(50), use_container_width=True)

            # Show summary stats
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

            # Store in session state for other pages
            st.session_state["current_dataset"] = df
            st.session_state["current_dataset_name"] = dataset_name
            st.session_state["indicator_names"] = {
                code: all_indicators.get(code, code) for code in selected_codes
            }

        except Exception as e:
            st.error(f"Download failed: {e}")
