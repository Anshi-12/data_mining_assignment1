"""Phase 6 Modeling page. Presentation only; fitting lives in modeling/."""
from __future__ import annotations

import logging
import pandas as pd
import streamlit as st

from crispdm_studio.state import invalidate_from

from crispdm_studio.modeling import run_modeling
from crispdm_studio.preparation import prepare_dataset
from crispdm_studio.state import (DATASET_KEY, EVALUATION_RESULT_KEY, MODEL_TARGET_KEY, MODELING_RESULT_KEY, PREPARATION_RESULT_KEY, REPORT_RESULT_KEY, UNDERSTANDING_RESULT_KEY)
from crispdm_studio.understanding import profile_dataset
from crispdm_studio.ui.messages import render_info

logger = logging.getLogger(__name__)


def _cv_text(model, metric: str) -> str:
    item = next((m for m in model.cv_metrics if m.name == metric), None)
    return "—" if item is None else f"{item.mean:.3f} ± {item.std:.3f}"


def render() -> None:
    st.title("Modeling + Baseline")
    dataset = st.session_state.get(DATASET_KEY)
    if dataset is None:
        render_info("Upload and validate a CSV before opening Modeling.")
        return

    try:
        understanding = profile_dataset(dataset)
        preparation = prepare_dataset(dataset.dataframe, understanding)
        st.session_state[UNDERSTANDING_RESULT_KEY] = understanding
        st.session_state[PREPARATION_RESULT_KEY] = preparation
    except Exception:
        logger.exception("Unable to prepare modeling inputs")
        invalidate_from(st.session_state, MODELING_RESULT_KEY)
        st.error("We couldn't safely prepare this dataset for modeling. No model was fit.", icon="⚠️")
        return

    frame = preparation.structural_dataframe
    candidates = {item.column: item for item in understanding.target_candidates if item.column in frame.columns}

    st.info(
        "**Leakage-safe approach:** the raw structural rows are split first. Each estimator is wrapped with a fresh copy "
        "of the Phase 3 preprocessor in one sklearn Pipeline. Imputation, IQR bounds, scaling, and categorical vocabularies "
        "are therefore learned from training rows only; cross-validation refits the whole pipeline inside each fold. "
        "The original Phase 3 preprocessing spec remains unfitted.",
        icon="🔒",
    )

    options = [None, *frame.columns.tolist()]
    target = st.selectbox(
        "Confirmed target",
        options,
        index=0,
        format_func=lambda value: "— Select a target —" if value is None else (f"{value} ★ suggested" if value in candidates else str(value)),
        help="Phase 2 candidates are suggestions only. Nothing is selected automatically.",
        key=MODEL_TARGET_KEY,
    )

    if candidates:
        st.caption("Phase 2 suggestions: " + ", ".join(f"{name} ({item.task_type.value}, {item.score:.0%})" for name, item in candidates.items()))
    else:
        st.caption("Phase 2 did not infer a strong target candidate; you may still explicitly choose a defensible target if you know the domain.")

    if target is None:
        st.session_state[MODELING_RESULT_KEY] = None
        st.session_state[EVALUATION_RESULT_KEY] = None
        st.session_state[REPORT_RESULT_KEY] = None
        st.warning("No target confirmed. Supervised modeling is intentionally skipped until you explicitly select one.", icon="⏭️")
        return

    try:
        with st.status("Splitting first, then fitting leakage-safe pipelines and cross-validation folds…"):
            result = run_modeling(frame, understanding, preparation, target)
            st.session_state[MODELING_RESULT_KEY] = result
            st.session_state[EVALUATION_RESULT_KEY] = None
            st.session_state[REPORT_RESULT_KEY] = None
    except Exception:
        logger.exception("Unexpected modeling failure")
        invalidate_from(st.session_state, MODELING_RESULT_KEY)
        st.error("We couldn't complete supervised modeling safely. No partial model result is being presented as valid.", icon="⚠️")
        return

    if getattr(result, "resource_notes", ()):
        for note in result.resource_notes:
            st.info(note, icon="ℹ️")
    if getattr(result, "model_failures", ()):
        st.warning("Some candidate models failed safely and were excluded from comparison.", icon="⚠️")
        for failure in result.model_failures:
            st.write(f"- {failure}")

    st.subheader("Target assessment")
    if result.task_type is not None:
        st.metric("Detected task", result.task_type.value.title())
    st.write(result.target_assessment.reason)

    if result.skipped:
        st.warning("Supervised modeling was skipped rather than forcing an indefensible model.", icon="⏭️")
        st.write(result.skip_reason)
        return

    assert result.split is not None
    cols = st.columns(5)
    cols[0].metric("Training rows", result.split.train_rows)
    cols[1].metric("Test rows", result.split.test_rows)
    cols[2].metric("Predictors", len(result.feature_columns))
    cols[3].metric("CV folds", result.cv_folds if result.cv_folds else "Skipped")
    cols[4].metric("Split", "Stratified" if result.split.stratified else "Random")
    st.caption(result.leakage_note)

    st.subheader("Baseline vs candidate models")
    rows = []
    if result.task_type.value == "classification":
        metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    else:
        metrics = ["mae", "rmse", "r2"]
    for model in result.model_results:
        row = {"Model": model.name, "Role": "Baseline" if model.is_baseline else "Candidate"}
        for metric in metrics:
            row[f"Test {metric}"] = model.test_metrics.get(metric)
            row[f"CV {metric}"] = _cv_text(model, metric)
        row["Overfitting flag"] = "Yes" if model.overfitting_flag else "No"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Train/test divergence")
    flagged = [m for m in result.model_results if m.overfitting_flag]
    if not flagged:
        st.success("No configured train/test divergence threshold was exceeded.", icon="✅")
    else:
        for model in flagged:
            st.warning(f"**{model.name}:** {model.overfitting_reason}")

    with st.expander("Per-model stored metrics", expanded=False):
        for model in result.model_results:
            st.markdown(f"**{model.name}**")
            st.json({"train": model.train_metrics, "test": model.test_metrics, "cv": {m.name: {"mean": m.mean, "std": m.std} for m in model.cv_metrics}})

    st.caption("The fitted Pipeline objects and metric structures are retained in ModelingResult for Phase 7 evaluation and Phase 8 reporting; they are not refit by this page.")
