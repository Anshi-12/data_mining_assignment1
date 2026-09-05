"""Phase 7 Evaluation page. Presentation only; it never refits Phase 6 models."""
from __future__ import annotations

import logging
import pandas as pd
import streamlit as st

from crispdm_studio.state import invalidate_from

from crispdm_studio.clustering import run_clustering
from crispdm_studio.eda import run_eda
from crispdm_studio.evaluation import evaluate_modeling
from crispdm_studio.preparation import prepare_dataset
from crispdm_studio.state import (
    CLUSTERING_RESULT_KEY, DATASET_KEY, EDA_RESULT_KEY, EVALUATION_RESULT_KEY,
    MODELING_RESULT_KEY, PREPARATION_RESULT_KEY, REPORT_RESULT_KEY, UNDERSTANDING_RESULT_KEY,
)
from crispdm_studio.understanding import profile_dataset
from crispdm_studio.ui.messages import render_info

logger = logging.getLogger(__name__)


def render() -> None:
    st.title("Evaluation + Recommendation")
    dataset = st.session_state.get(DATASET_KEY)
    if dataset is None:
        render_info("Upload and validate a CSV before opening Evaluation.")
        return
    modeling = st.session_state.get(MODELING_RESULT_KEY)
    if modeling is None:
        st.info("Open Modeling, explicitly select a target, and run Phase 6 first. Evaluation reuses that stored fitted ModelingResult and never retrains it.", icon="ℹ️")
        return

    st.info(
        "**Read-only evaluation:** Phase 7 reuses the already-fitted Phase 6 pipelines. It does not call `.fit()` on an estimator or preprocessor. "
        "Permutation importance and error diagnostics are calculated only against the stored held-out test rows.",
        icon="🔒",
    )
    try:
        understanding = st.session_state.get(UNDERSTANDING_RESULT_KEY) or profile_dataset(dataset)
        preparation = st.session_state.get(PREPARATION_RESULT_KEY) or prepare_dataset(dataset.dataframe, understanding)
        eda = st.session_state.get(EDA_RESULT_KEY) or run_eda(preparation.structural_dataframe, understanding)
        clustering = st.session_state.get(CLUSTERING_RESULT_KEY) or run_clustering(preparation.structural_dataframe, understanding)
        st.session_state[UNDERSTANDING_RESULT_KEY] = understanding
        st.session_state[PREPARATION_RESULT_KEY] = preparation
        st.session_state[EDA_RESULT_KEY] = eda
        st.session_state[CLUSTERING_RESULT_KEY] = clustering
        with st.status("Evaluating stored holdout evidence without retraining…"):
            result = evaluate_modeling(modeling, eda, clustering, understanding)
        st.session_state[EVALUATION_RESULT_KEY] = result
        st.session_state[REPORT_RESULT_KEY] = None
    except Exception:
        logger.exception("Unexpected evaluation failure")
        invalidate_from(st.session_state, EVALUATION_RESULT_KEY)
        st.error("We couldn't complete evaluation safely. The fitted Phase 6 models were not retrained or modified.", icon="⚠️")
        return

    st.subheader("Reasoned model selection")
    if result.skipped:
        st.warning("No defensible predictive model is recommended from this run.", icon="⏭️")
        st.write(result.skip_reason)
    else:
        st.success(f"Selected: **{result.selected_model_name}**")
        st.write(result.selection_reason)

    st.subheader("Improvement over Dummy baseline")
    if result.baseline_improvements:
        st.dataframe(pd.DataFrame([{
            "Model": x.model_name,
            "Metric": x.metric,
            "Baseline": x.baseline_value,
            "Model value": x.model_value,
            "Absolute improvement": x.absolute_improvement,
            "Relative improvement": x.relative_improvement,
            "Beats baseline": x.beats_baseline,
        } for x in result.baseline_improvements]), hide_index=True, use_container_width=True)
    else:
        st.caption("No baseline comparison is available.")

    st.subheader("Stability, overfitting, imbalance, and complexity")
    if result.selection_scores:
        st.dataframe(pd.DataFrame([{
            "Model": x.model_name,
            "Test improvement": x.test_improvement,
            "CV improvement": x.cv_improvement,
            "CV std": x.cv_std,
            "Overfitting penalty": x.overfitting_penalty,
            "Complexity penalty": x.complexity_penalty,
            "Risk-adjusted score": x.composite_score,
            "Reason": x.reason,
        } for x in result.selection_scores]), hide_index=True, use_container_width=True)
    st.write(result.imbalance.message)

    if result.permutation_importance:
        st.subheader("Held-out permutation importance")
        st.dataframe(pd.DataFrame([{
            "Feature": x.feature,
            "Importance mean": x.importance_mean,
            "Importance std": x.importance_std,
        } for x in result.permutation_importance]), hide_index=True, use_container_width=True)

    for chart in result.charts:
        st.subheader(chart.title)
        st.plotly_chart(chart.figure, use_container_width=True, key=chart.chart_id)

    st.subheader("Plain-English recommendation")
    recommendation = result.recommendation
    sections = [
        ("What we learned", recommendation.what_we_learned),
        ("How reliable it appears", recommendation.reliability),
        ("What action is reasonable", recommendation.reasonable_action),
        ("What limitations matter", recommendation.limitations),
        ("What to do next", recommendation.next_steps),
    ]
    for title, text in sections:
        st.markdown(f"**{title}**")
        st.write(text)

    st.caption("The charts and recommendation above are stored in EvaluationResult for Phase 8; report generation can reuse them without refitting or recomputing model evaluation.")
