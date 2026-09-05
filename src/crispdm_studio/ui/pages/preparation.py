"""Phase 3 Preparation page. Presentation only; transformation logic lives in preparation/."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from crispdm_studio.state import invalidate_from

from crispdm_studio.preparation import DecisionStage, prepare_dataset
from crispdm_studio.state import DATASET_KEY, PREPARATION_RESULT_KEY, UNDERSTANDING_RESULT_KEY
from crispdm_studio.understanding import profile_dataset
from crispdm_studio.ui.messages import render_info

logger = logging.getLogger(__name__)


def _decision_frame(decisions) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Status": "Applied now — structural" if item.stage is DecisionStage.APPLIED_NOW else "Deferred — learned",
                "Decision": item.action,
                "Why": item.reason,
                "Rows affected": item.rows_affected,
                "Columns affected": item.columns_affected,
                "Columns": ", ".join(item.columns) if item.columns else "—",
                "Before": item.before or "—",
                "After": item.after or "—",
            }
            for item in decisions
        ]
    )


def render() -> None:
    st.title("Data Preparation")
    dataset = st.session_state.get(DATASET_KEY)
    if dataset is None:
        render_info("Upload and validate a CSV on the Upload page before opening Preparation.")
        return

    try:
        with st.status("Building leakage-safe preparation decisions…"):
            understanding = profile_dataset(dataset)
            result = prepare_dataset(dataset.dataframe, understanding)
            st.session_state[UNDERSTANDING_RESULT_KEY] = understanding
            st.session_state[PREPARATION_RESULT_KEY] = result
    except Exception:
        logger.exception("Unexpected Preparation analysis failure")
        invalidate_from(st.session_state, PREPARATION_RESULT_KEY)
        st.error(
            "We couldn't complete data preparation safely. The original uploaded data has not been modified; "
            "please return to Upload to inspect it or try another CSV.",
            icon="⚠️",
        )
        return

    st.warning(
        "Leakage boundary: only deterministic structural cleaning is applied to the full dataset here. "
        "Imputation, encoding, outlier bounds, and scaling remain UNFITTED and must be fit inside training folds in Phase 6.",
        icon="🛡️",
    )

    summary = result.summary
    metrics = st.columns(6)
    metrics[0].metric("Rows before", f"{summary.rows_before:,}")
    metrics[1].metric("Rows after", f"{summary.rows_after_structural:,}", f"-{summary.rows_removed:,}")
    metrics[2].metric("Columns before", f"{summary.columns_before:,}")
    metrics[3].metric("Columns after", f"{summary.columns_after_structural:,}", f"-{summary.columns_removed:,}")
    metrics[4].metric("Applied now", summary.applied_decisions)
    metrics[5].metric("Deferred", summary.deferred_decisions)

    st.subheader("Auditable preparation decisions")
    st.dataframe(_decision_frame(result.decisions), hide_index=True, use_container_width=True, height=480)

    applied_tab, deferred_tab = st.tabs(["Applied now — structural", "Deferred to modeling — learned"])
    with applied_tab:
        st.caption(
            "These operations do not estimate dataset statistics or vocabularies and are safe to apply before splitting."
        )
        st.dataframe(_decision_frame(result.applied_decisions), hide_index=True, use_container_width=True)
    with deferred_tab:
        st.caption(
            "These operations learn medians, modes, category sets/frequencies, IQR fences, means, or standard deviations. "
            "They are specifications only in Phase 3 and are not fit on the full dataframe."
        )
        st.dataframe(_decision_frame(result.deferred_decisions), hide_index=True, use_container_width=True)

    st.subheader("Deferred sklearn preprocessing specification")
    spec = result.preprocessing_spec
    spec_frame = pd.DataFrame(
        [
            {"Role": "Numeric", "Columns": ", ".join(spec.numeric_columns) or "—", "Strategy": "median impute → IQR clip → StandardScaler"},
            {"Role": "Categorical", "Columns": ", ".join(spec.categorical_columns) or "—", "Strategy": "most-frequent impute → OneHotEncoder"},
            {"Role": "High-cardinality", "Columns": ", ".join(spec.high_cardinality_columns) or "—", "Strategy": f"group/cap within OneHotEncoder (max {spec.encoding.max_categories}, min frequency {spec.encoding.min_frequency:.0%})"},
            {"Role": "Excluded from generic baseline", "Columns": ", ".join(spec.excluded_columns) or "—", "Strategy": "preserved structurally; not transformed until explicit feature engineering exists"},
        ]
    )
    st.dataframe(spec_frame, hide_index=True, use_container_width=True)
    st.info(
        "`PreprocessingSpec.build_transformer()` returns a new, unfitted sklearn ColumnTransformer. "
        "Phase 6 will place it inside each model Pipeline so cross-validation fits transformations on training folds only.",
        icon="ℹ️",
    )

    st.subheader("Structurally cleaned preview")
    st.dataframe(result.structural_dataframe.head(100), use_container_width=True, height=350)
    st.caption(
        "Missing values remain visible by design. Seeing them here confirms that learned imputation has not been fit globally."
    )
