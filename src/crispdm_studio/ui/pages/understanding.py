"""Phase 2 Understanding page. Presentation only; analysis lives in understanding/."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from crispdm_studio.state import invalidate_from

from crispdm_studio.state import DATASET_KEY, UNDERSTANDING_RESULT_KEY
from crispdm_studio.understanding import profile_dataset
from crispdm_studio.understanding.quality import WarningSeverity
from crispdm_studio.ui.messages import render_info

logger = logging.getLogger(__name__)


def _format_number(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.4g}"


def render() -> None:
    st.title("Business & Data Understanding")
    dataset = st.session_state.get(DATASET_KEY)
    if dataset is None:
        render_info("Upload and validate a CSV on the Upload page before opening Understanding.")
        return

    try:
        with st.status("Profiling the dataset and inferring analytical structure…"):
            result = profile_dataset(dataset)
            st.session_state[UNDERSTANDING_RESULT_KEY] = result
    except Exception:
        logger.exception("Unexpected Understanding analysis failure")
        invalidate_from(st.session_state, UNDERSTANDING_RESULT_KEY)
        st.error(
            "We couldn't complete the dataset profile safely. The uploaded file is still available; "
            "please return to Upload to inspect it or try another CSV.",
            icon="⚠️",
        )
        return

    profile = result.dataset
    metric_cols = st.columns(5)
    metric_cols[0].metric("Rows", f"{profile.rows:,}")
    metric_cols[1].metric("Columns", f"{profile.columns:,}")
    metric_cols[2].metric("Missing cells", f"{profile.missing_cells:,}", f"{profile.missing_cell_ratio:.1%}")
    metric_cols[3].metric("Duplicate rows", f"{profile.duplicate_rows:,}", f"{profile.duplicate_row_ratio:.1%}")
    metric_cols[4].metric("Possible timestamps", f"{len(profile.possible_timestamp_columns):,}")

    st.subheader("Generated analytical objective")
    st.caption("KNOWN facts are computed from the uploaded data. INFERRED objectives are hypotheses, not supplied business context.")
    left, right = st.columns(2)
    with left:
        st.markdown("**KNOWN — observed from the file**")
        for fact in result.objective.known_facts:
            st.markdown(f"- {fact}")
    with right:
        st.markdown("**INFERRED — proposed analytical objectives**")
        for objective in result.objective.inferred_objectives:
            st.markdown(f"- {objective}")
    st.info(result.objective.honesty_note, icon="ℹ️")

    st.subheader("Feature-type overview")
    type_frame = pd.DataFrame(
        [
            {"Feature type": feature_type, "Columns": count}
            for feature_type, count in sorted(profile.feature_type_counts.items())
        ]
    )
    st.dataframe(type_frame, hide_index=True, use_container_width=True)

    st.subheader("Per-column profile")
    profile_rows = []
    for column in result.columns:
        profile_rows.append(
            {
                "Column": column.name,
                "Inferred type": column.feature_type.value,
                "Storage dtype": column.storage_dtype,
                "Confidence": f"{column.type_confidence:.0%}",
                "Missing": f"{column.missing:,} ({column.missing_ratio:.1%})",
                "Unique": f"{column.unique:,} ({column.uniqueness_ratio:.1%})",
                "Constant": "Yes" if column.is_constant else "No",
                "Near-constant": "Yes" if column.is_near_constant else "No",
                "Possible timestamp": "Yes" if column.possible_timestamp else "No",
                "Observed summary": column.observed_summary,
                "Type inference basis": column.type_reason,
            }
        )
    st.dataframe(pd.DataFrame(profile_rows), hide_index=True, use_container_width=True, height=430)

    with st.expander("Distribution details", expanded=False):
        for column in result.columns:
            st.markdown(f"**{column.name} — {column.feature_type.value}**")
            if column.numeric_summary is not None:
                summary = column.numeric_summary
                stats = pd.DataFrame(
                    {
                        "Statistic": ["Mean", "Std. dev.", "Min", "25%", "Median", "75%", "Max", "Skewness"],
                        "Value": [
                            _format_number(summary.mean),
                            _format_number(summary.std),
                            _format_number(summary.minimum),
                            _format_number(summary.q25),
                            _format_number(summary.median),
                            _format_number(summary.q75),
                            _format_number(summary.maximum),
                            _format_number(summary.skewness),
                        ],
                    }
                )
                st.dataframe(stats, hide_index=True, use_container_width=True)
            if column.top_values:
                top = pd.DataFrame(
                    [
                        {"Value": item.label, "Count": item.count, "Share": f"{item.share:.1%}"}
                        for item in column.top_values
                    ]
                )
                st.caption("Top observed non-missing values")
                st.dataframe(top, hide_index=True, use_container_width=True)

    st.subheader("Identifiers & possible leakage")
    if result.identifiers:
        st.write("**Identifier-like columns:** " + ", ".join(f"`{name}`" for name in result.identifiers))
    else:
        st.write("No identifier-like columns were detected by the current structural heuristics.")

    if result.leakage_flags:
        leakage_frame = pd.DataFrame(
            [{"Column": item.column, "Risk": item.risk, "Why review it": item.reason} for item in result.leakage_flags]
        )
        st.dataframe(leakage_frame, hide_index=True, use_container_width=True)
        st.caption("Leakage flags are review prompts, not proof that a feature leaks future information.")
    else:
        st.success("No obvious structural leakage signals were detected.", icon="✅")

    st.subheader("Candidate supervised-learning targets")
    if result.target_candidates:
        target_frame = pd.DataFrame(
            [
                {
                    "Candidate": item.column,
                    "Likely task": item.task_type.value,
                    "Heuristic score": f"{item.score:.0%}",
                    "Why plausible": " ".join(item.reasons),
                    "Caveats": " ".join(item.caveats) or "—",
                }
                for item in result.target_candidates
            ]
        )
        st.dataframe(target_frame, hide_index=True, use_container_width=True)
        st.caption("No target is selected in Phase 2. A later phase will require user confirmation before supervised modeling.")
    else:
        st.info("No defensible supervised-learning target candidate was inferred from structure alone.", icon="ℹ️")

    st.subheader("Data-quality warnings")
    if not result.quality_warnings:
        st.success("No Phase 2 quality warnings were triggered.", icon="✅")
    else:
        for warning in result.quality_warnings:
            if warning.severity is WarningSeverity.HIGH:
                st.error(warning.message, icon="🚨")
            elif warning.severity is WarningSeverity.WARNING:
                st.warning(warning.message, icon="⚠️")
            else:
                st.info(warning.message, icon="ℹ️")
