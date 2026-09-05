"""Streamlit entrypoint for CRISP-DM Studio with a final application-wide safety boundary."""
import logging
import streamlit as st

from crispdm_studio.config import CONFIG
from crispdm_studio.logging_config import configure_logging
from crispdm_studio.state import (
    CLUSTERING_RESULT_KEY, EDA_RESULT_KEY, EVALUATION_RESULT_KEY, MODELING_RESULT_KEY,
    PREPARATION_RESULT_KEY, REPORT_RESULT_KEY, UNDERSTANDING_RESULT_KEY,
    initialize_state, invalidate_from,
)
from crispdm_studio.ui.pages.preparation import render as render_preparation
from crispdm_studio.ui.pages.eda import render as render_eda
from crispdm_studio.ui.pages.clustering import render as render_clustering
from crispdm_studio.ui.pages.modeling import render as render_modeling
from crispdm_studio.ui.pages.evaluation import render as render_evaluation
from crispdm_studio.ui.pages.report import render as render_report
from crispdm_studio.ui.pages.understanding import render as render_understanding
from crispdm_studio.ui.pages.upload import render as render_upload
from crispdm_studio.ui.theme import apply_theme

configure_logging()
logger = logging.getLogger(__name__)
st.set_page_config(page_title=CONFIG.app_name, page_icon="📊", layout="wide")
initialize_state(st.session_state)
apply_theme()

page = st.sidebar.radio(
    "CRISP-DM phase",
    ["Upload", "Understanding", "Preparation", "EDA", "Clustering", "Modeling", "Evaluation", "Report"],
    help=(
        "Each page has a defensive failure boundary. Expensive phases use deterministic resource caps; "
        "a new upload invalidates all downstream results so stale analyses cannot leak across datasets."
    ),
)

routes = {
    "Upload": (render_upload, None),
    "Understanding": (render_understanding, UNDERSTANDING_RESULT_KEY),
    "Preparation": (render_preparation, PREPARATION_RESULT_KEY),
    "EDA": (render_eda, EDA_RESULT_KEY),
    "Clustering": (render_clustering, CLUSTERING_RESULT_KEY),
    "Modeling": (render_modeling, MODELING_RESULT_KEY),
    "Evaluation": (render_evaluation, EVALUATION_RESULT_KEY),
    "Report": (render_report, REPORT_RESULT_KEY),
}
renderer, result_key = routes[page]
try:
    renderer()
except Exception:
    # Last-resort boundary: individual pages already handle expected failures, but
    # a UI bug or third-party renderer error must still never expose a traceback.
    logger.exception("Unhandled %s page failure", page)
    if result_key:
        invalidate_from(st.session_state, result_key)
    st.error(
        f"We couldn't complete {page} safely. No stale or partial {page.lower()} result is being shown. "
        "You can return to an earlier completed stage or try a different dataset.",
        icon="⚠️",
    )
