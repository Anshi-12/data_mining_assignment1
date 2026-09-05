"""Phase 1 upload page. Contains presentation logic only."""

import logging

import streamlit as st

from crispdm_studio.config import CONFIG
from crispdm_studio.exceptions import CrispDMError
from crispdm_studio.ingestion import ingest_csv
from crispdm_studio.state import clear_dataset, set_dataset, set_error
from crispdm_studio.ui.components import render_dataset_overview
from crispdm_studio.ui.messages import render_error, render_info

logger = logging.getLogger(__name__)


def render() -> None:
    st.title("CRISP-DM Studio")
    st.write(
        "Upload a CSV to validate it and inspect a safe dataset overview. "
        "Later phases will build the full CRISP-DM pipeline on this same ingestion layer."
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        accept_multiple_files=False,
        help=f"Maximum file size for analysis: {CONFIG.max_upload_mb} MB.",
    )

    if uploaded is None:
        clear_dataset(st.session_state)
        render_info("Drag and drop a CSV here, or use Browse files to choose one.")
        return

    try:
        with st.spinner("Validating and reading the CSV…"):
            dataset = ingest_csv(
                filename=uploaded.name,
                mime_type=uploaded.type or "",
                data=uploaded.getvalue(),
                config=CONFIG,
            )
        set_dataset(st.session_state, dataset)
        render_dataset_overview(dataset, preview_rows=CONFIG.preview_rows)
    except CrispDMError as exc:
        logger.info("User-facing upload validation error: %s", exc.user_message)
        set_error(st.session_state, exc.user_message)
        render_error(exc.user_message)
    except Exception:
        # UI boundary is intentionally defensive. Internal details go only to logs.
        logger.exception("Unexpected upload-page error")
        message = "We couldn't analyze this file safely. Please check the CSV and try again."
        set_error(st.session_state, message)
        render_error(message)
