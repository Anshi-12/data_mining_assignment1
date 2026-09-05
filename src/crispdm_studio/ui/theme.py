"""Small UI styling helpers."""

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem;}
        div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: .8rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
