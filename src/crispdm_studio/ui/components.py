"""Reusable Streamlit presentation components."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from crispdm_studio.models import IngestedDataset


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def render_dataset_overview(dataset: IngestedDataset, preview_rows: int = 100) -> None:
    overview = dataset.overview
    st.success("CSV validated successfully. The dataset is ready for CRISP-DM analysis.", icon="✅")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Rows", f"{overview.rows:,}")
    metric_cols[1].metric("Columns", f"{overview.columns:,}")
    metric_cols[2].metric("Missing cells", f"{overview.missing_cells:,}")
    metric_cols[3].metric("Duplicate rows", f"{overview.duplicate_rows:,}")
    metric_cols[4].metric("Memory", _human_bytes(overview.memory_bytes))

    with st.expander("File and parsing details", expanded=False):
        details = pd.DataFrame(
            {
                "Property": ["Filename", "File size", "MIME type", "Encoding", "Delimiter"],
                "Value": [
                    dataset.metadata.filename,
                    _human_bytes(dataset.metadata.size_bytes),
                    dataset.metadata.mime_type,
                    dataset.metadata.encoding,
                    repr(dataset.metadata.delimiter),
                ],
            }
        )
        st.dataframe(details, hide_index=True, use_container_width=True)

    if dataset.warnings:
        for warning in dataset.warnings:
            st.warning(warning, icon="⚠️")

    st.subheader("Dataset preview")
    st.caption(f"Showing up to the first {min(preview_rows, overview.rows):,} rows.")
    st.dataframe(dataset.dataframe.head(preview_rows), use_container_width=True, height=420)

    st.subheader("Column overview")
    column_summary = pd.DataFrame(
        {
            "Column": dataset.dataframe.columns,
            "Type": dataset.dataframe.dtypes.astype(str).values,
            "Non-null": dataset.dataframe.notna().sum().values,
            "Missing": dataset.dataframe.isna().sum().values,
            "Unique": [dataset.dataframe[col].nunique(dropna=True) for col in dataset.dataframe.columns],
        }
    )
    st.dataframe(column_summary, hide_index=True, use_container_width=True)
