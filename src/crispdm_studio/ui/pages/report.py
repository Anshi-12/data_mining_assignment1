"""Phase 8 report page. It renders only already-computed result objects."""
from __future__ import annotations

import logging
import streamlit as st

from crispdm_studio.state import invalidate_from

from crispdm_studio.reporting import build_report
from crispdm_studio.state import (
    CLUSTERING_RESULT_KEY,
    DATASET_KEY,
    EDA_RESULT_KEY,
    EVALUATION_RESULT_KEY,
    MODELING_RESULT_KEY,
    PREPARATION_RESULT_KEY,
    REPORT_RESULT_KEY,
    UNDERSTANDING_RESULT_KEY,
)
from crispdm_studio.ui.messages import render_info

logger = logging.getLogger(__name__)


def render() -> None:
    st.title("CRISP-DM Report")
    if st.session_state.get(DATASET_KEY) is None:
        render_info("Upload and validate a CSV before opening Report.")
        return

    keys = (
        (UNDERSTANDING_RESULT_KEY, "Understanding"),
        (PREPARATION_RESULT_KEY, "Preparation"),
        (EDA_RESULT_KEY, "EDA"),
        (CLUSTERING_RESULT_KEY, "Clustering"),
        (MODELING_RESULT_KEY, "Modeling"),
        (EVALUATION_RESULT_KEY, "Evaluation"),
    )
    missing = [label for key, label in keys if st.session_state.get(key) is None]
    if missing:
        st.info(
            "The report is enabled only after the pipeline results exist. Run these pages first: "
            + ", ".join(missing)
            + ". The report page will not recompute them.",
            icon="ℹ️",
        )
        st.button("Download HTML Report", disabled=True)
        st.button("Download PDF Report", disabled=True)
        return

    st.info(
        "**Analyze once, render many ways:** this page consumes the stored results, findings, decisions, fitted-model evaluation, "
        "recommendation, and the exact stored chart objects. It does not rerun EDA/clustering, recompute evaluation, or refit a model/preprocessor.",
        icon="📄",
    )

    report = st.session_state.get(REPORT_RESULT_KEY)
    if report is None:
        try:
            with st.status("Rendering the canonical HTML and robust PDF export…"):
                report = build_report(
                    st.session_state[UNDERSTANDING_RESULT_KEY],
                    st.session_state[PREPARATION_RESULT_KEY],
                    st.session_state[EDA_RESULT_KEY],
                    st.session_state[CLUSTERING_RESULT_KEY],
                    st.session_state[MODELING_RESULT_KEY],
                    st.session_state[EVALUATION_RESULT_KEY],
                )
                st.session_state[REPORT_RESULT_KEY] = report
        except Exception:
            logger.exception("Unexpected report generation failure")
            invalidate_from(st.session_state, REPORT_RESULT_KEY)
            st.error(
                "We couldn't render the report safely. The previously computed analysis results remain unchanged.",
                icon="⚠️",
            )
            return

    st.subheader("What's included")
    st.write(
        "Executive summary; Business Understanding; Data Understanding; Data Preparation; EDA; Clustering; Predictive Modeling; "
        "Model Comparison; Evaluation; Limitations; Recommendation; and Technical appendix. Skipped analyses are included explicitly."
    )

    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download HTML Report",
            data=report.html_bytes,
            file_name="crispdm-report.html",
            mime="text/html",
            use_container_width=True,
        )
        st.caption("Standalone interactive HTML. No Chrome, WeasyPrint, or system PDF dependency is required.")
    with right:
        st.download_button(
            "Download PDF Report",
            data=report.pdf_bytes,
            file_name="crispdm-report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.caption(f"PDF fidelity: {report.pdf.fidelity}.")

    if report.pdf.notices:
        st.warning("The PDF was still produced, but one or more optional rendering prerequisites were unavailable.", icon="⚠️")
        seen: set[tuple[str, str | None]] = set()
        for notice in report.pdf.notices:
            key = (notice.message, notice.install_hint)
            if key in seen:
                continue
            seen.add(key)
            st.write(f"**{notice.component}:** {notice.message}")
            if notice.install_hint:
                st.code(notice.install_hint, language=None)
    else:
        st.success("Full-fidelity PDF path succeeded, including static chart rendering and styled HTML→PDF conversion.", icon="✅")

    st.caption(
        f"PDF static charts rendered: {report.pdf.static_charts_rendered}; placeholders: {report.pdf.static_charts_placeholder}; "
        f"WeasyPrint succeeded: {report.pdf.weasyprint_succeeded}."
    )
