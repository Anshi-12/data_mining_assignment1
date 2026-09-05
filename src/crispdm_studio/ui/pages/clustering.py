"""Phase 5 Clustering page. Presentation only."""
from __future__ import annotations

import logging
import pandas as pd
import streamlit as st

from crispdm_studio.state import invalidate_from

from crispdm_studio.clustering import run_clustering
from crispdm_studio.preparation import prepare_dataset
from crispdm_studio.state import CLUSTERING_RESULT_KEY, DATASET_KEY, PREPARATION_RESULT_KEY, UNDERSTANDING_RESULT_KEY
from crispdm_studio.understanding import profile_dataset
from crispdm_studio.ui.messages import render_info

logger = logging.getLogger(__name__)


def render() -> None:
    st.title("Clustering")
    dataset = st.session_state.get(DATASET_KEY)
    if dataset is None:
        render_info("Upload and validate a CSV on the Upload page before opening Clustering.")
        return
    try:
        with st.status("Searching for meaningful cluster structure…"):
            understanding = profile_dataset(dataset)
            preparation = prepare_dataset(dataset.dataframe, understanding)
            result = run_clustering(preparation.structural_dataframe, understanding)
            st.session_state[UNDERSTANDING_RESULT_KEY] = understanding
            st.session_state[PREPARATION_RESULT_KEY] = preparation
            st.session_state[CLUSTERING_RESULT_KEY] = result
    except Exception:
        logger.exception("Unexpected clustering failure")
        invalidate_from(st.session_state, CLUSTERING_RESULT_KEY)
        st.error("We couldn't complete clustering safely. No supervised transformer or train/test split was touched.", icon="⚠️")
        return

    if getattr(result, "resource_notes", ()):
        for note in result.resource_notes:
            st.info(note, icon="ℹ️")

    st.info(
        "Clustering is unsupervised. Its own imputation, encoding, and scaling are fit on the complete unlabeled structural dataframe. "
        "This is separate from Phase 6 supervised modeling; the Phase 3 deferred supervised transformer remains unfitted.",
        icon="ℹ️",
    )

    spec = result.feature_spec
    st.caption(
        f"Eligible features: {len(spec.eligible_columns)} ({', '.join(spec.eligible_columns) or 'none'}). "
        f"Excluded: {', '.join(spec.excluded_columns) or 'none'}."
    )

    if result.skipped:
        st.warning("Clustering was skipped rather than inventing segments.", icon="⏭️")
        for item in result.skipped_analyses:
            st.write(f"**{item.analysis}:** {item.reason}")
        if result.candidates:
            st.subheader("Diagnostics evaluated before skipping")
            st.dataframe(pd.DataFrame([{
                "k": c.k, "Silhouette": c.silhouette, "Inertia": c.inertia,
                "Calinski-Harabasz": c.calinski_harabasz,
                "Davies-Bouldin": c.davies_bouldin,
                "Smallest cluster": c.min_cluster_size,
            } for c in result.candidates]), hide_index=True, use_container_width=True)
            for chart in result.charts:
                st.plotly_chart(chart.figure, use_container_width=True, key=chart.chart_id)
        return

    cols = st.columns(4)
    cols[0].metric("Chosen k", result.chosen_k)
    cols[1].metric("Silhouette", f"{result.silhouette:.3f}" if result.silhouette is not None else "—")
    cols[2].metric("Rows clustered", f"{len(preparation.structural_dataframe):,}")
    cols[3].metric("Eligible features", len(spec.eligible_columns))
    st.success(result.selection_reason)

    st.subheader("Candidate diagnostics")
    st.dataframe(pd.DataFrame([{
        "k": c.k, "Silhouette": c.silhouette, "Inertia": c.inertia,
        "Calinski-Harabasz": c.calinski_harabasz,
        "Davies-Bouldin": c.davies_bouldin,
        "Smallest cluster": c.min_cluster_size,
        "Largest cluster": c.max_cluster_size,
    } for c in result.candidates]), hide_index=True, use_container_width=True)

    for chart in result.charts:
        st.subheader(chart.title)
        st.plotly_chart(chart.figure, use_container_width=True, key=chart.chart_id)

    st.subheader("Per-cluster profiles")
    rows = []
    for p in result.profiles:
        row = {"Cluster": p.cluster, "Rows": p.size, "Share": p.share}
        row.update({f"Median {k}": v for k, v in p.numeric_medians.items()})
        row.update({f"Mode {k}": v for k, v in p.categorical_modes.items()})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Computed segment descriptions")
    for description in result.segment_descriptions:
        st.markdown(f"**{description.title}**")
        st.write(description.text)
    st.caption("Descriptions above are generated from the stored cluster sizes, medians, and modes; no canned segment personas are used.")
