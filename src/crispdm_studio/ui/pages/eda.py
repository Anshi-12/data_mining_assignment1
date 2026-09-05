"""Phase 4 EDA page. Presentation only; all analysis lives in eda/."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from crispdm_studio.state import invalidate_from

from crispdm_studio.eda import run_eda
from crispdm_studio.preparation import prepare_dataset
from crispdm_studio.state import DATASET_KEY, EDA_RESULT_KEY, PREPARATION_RESULT_KEY, UNDERSTANDING_RESULT_KEY
from crispdm_studio.understanding import profile_dataset
from crispdm_studio.ui.messages import render_info

logger = logging.getLogger(__name__)


def _render_findings(result) -> None:
    st.subheader("Computed findings")
    if not result.findings:
        st.info("No evidence-backed findings met the current descriptive thresholds.", icon="ℹ️")
        return
    for finding in result.findings:
        with st.container(border=True):
            st.markdown(f"**{finding.title}**")
            st.write(finding.text)
            if finding.metrics:
                st.caption(", ".join(f"{key}={value}" for key, value in finding.metrics.items() if value is not None))


def _render_chart_section(result, section: str) -> None:
    charts = result.charts_for_section(section)
    st.subheader(section)
    if not charts:
        reason = next((item.reason for item in result.skipped_analyses if item.analysis == section), None)
        st.info(reason or "This analysis did not produce a chart for the current data.", icon="ℹ️")
        return
    for chart in charts:
        st.plotly_chart(chart.figure, use_container_width=True, config={"displaylogo": False})
        st.caption(chart.alt_text)


def render() -> None:
    st.title("Exploratory Data Analysis")
    dataset = st.session_state.get(DATASET_KEY)
    if dataset is None:
        render_info("Upload and validate a CSV on the Upload page before opening EDA.")
        return

    try:
        with st.status("Computing adaptive descriptive analysis…"):
            understanding = profile_dataset(dataset)
            preparation = prepare_dataset(dataset.dataframe, understanding)
            result = run_eda(preparation.structural_dataframe, understanding)
            st.session_state[UNDERSTANDING_RESULT_KEY] = understanding
            st.session_state[PREPARATION_RESULT_KEY] = preparation
            st.session_state[EDA_RESULT_KEY] = result
        if getattr(result, "resource_notes", ()):
            for note in result.resource_notes:
                st.info(note, icon="ℹ️")
    except Exception:
        logger.exception("Unexpected EDA analysis failure")
        invalidate_from(st.session_state, EDA_RESULT_KEY)
        st.error(
            "We couldn't complete exploratory analysis safely. The uploaded data has not been altered; "
            "please inspect the earlier phases or try another CSV.",
            icon="⚠️",
        )
        return

    st.info(
        "Phase 4 is descriptive only. It uses the structurally cleaned dataframe from Phase 3 and does not fit "
        "imputers, encoders, scalers, outlier bounds, models, or train/test splits.",
        icon="🔎",
    )

    metrics = st.columns(5)
    metrics[0].metric("Rows analyzed", f"{result.rows:,}")
    metrics[1].metric("Columns analyzed", f"{result.columns:,}")
    metrics[2].metric("Numeric columns", len(result.numeric_columns))
    metrics[3].metric("Categorical columns", len(result.categorical_columns))
    metrics[4].metric("Reusable charts", len(result.charts))

    _render_findings(result)

    if result.numeric_distributions:
        st.subheader("Numeric descriptive statistics")
        frame = pd.DataFrame(
            [
                {
                    "Column": item.column,
                    "Observed": item.count,
                    "Missing": f"{item.missing_ratio:.1%}",
                    "Mean": item.mean,
                    "Median": item.median,
                    "Std. dev.": item.std,
                    "5%": item.q05,
                    "25%": item.q25,
                    "75%": item.q75,
                    "95%": item.q95,
                    "Skewness": item.skewness,
                    "IQR outliers": f"{item.outlier_count:,} ({item.outlier_ratio:.1%})",
                }
                for item in result.numeric_distributions
            ]
        )
        st.dataframe(frame, hide_index=True, use_container_width=True)

    if result.categorical_distributions:
        st.subheader("Categorical distribution summary")
        frame = pd.DataFrame(
            [
                {
                    "Column": item.column,
                    "Observed": item.count,
                    "Missing": f"{item.missing_ratio:.1%}",
                    "Cardinality": item.cardinality,
                    "Dominant share": f"{item.dominant_share:.1%}" if item.dominant_share is not None else "—",
                    "Normalized entropy": f"{item.normalized_entropy:.3f}" if item.normalized_entropy is not None else "—",
                }
                for item in result.categorical_distributions
            ]
        )
        st.dataframe(frame, hide_index=True, use_container_width=True)

    for section in (
        "Missingness",
        "Numeric distributions",
        "Outlier diagnostics",
        "Categorical distributions",
        "Correlation analysis",
        "Numeric ↔ categorical relationships",
        "Categorical associations",
    ):
        _render_chart_section(result, section)

    st.subheader("Relationship metrics")
    if result.correlations:
        st.markdown("**Numeric correlations**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Left": item.left, "Right": item.right, "Pearson r": item.correlation, "Complete rows": item.observations}
                    for item in result.correlations
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    if result.numeric_categorical_relationships:
        st.markdown("**Numeric ↔ categorical effect sizes**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Numeric": item.numeric,
                        "Categorical": item.categorical,
                        "Eta-squared": item.eta_squared,
                        "Groups": item.groups,
                        "Complete rows": item.observations,
                    }
                    for item in result.numeric_categorical_relationships
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    if result.categorical_associations:
        st.markdown("**Categorical associations**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Left": item.left,
                        "Right": item.right,
                        "Cramér's V": item.cramers_v,
                        "Complete rows": item.observations,
                    }
                    for item in result.categorical_associations
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Analyses skipped by design")
    if not result.skipped_analyses:
        st.success("All Phase 4 analysis families were applicable to this dataset.", icon="✅")
    else:
        st.dataframe(
            pd.DataFrame(
                [{"Analysis": item.analysis, "Reason": item.reason} for item in result.skipped_analyses]
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Every chart above is stored in EDAResult as one reusable Plotly figure. The reporting layer can export the same "
        "object to HTML/PNG/SVG without recomputing EDA."
    )
