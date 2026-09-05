"""Friendly UI message helpers."""

import streamlit as st


def render_error(message: str) -> None:
    st.error(message, icon="🚫")


def render_info(message: str) -> None:
    st.info(message, icon="ℹ️")
