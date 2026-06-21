"""Helpers for managing Streamlit session state."""

import streamlit as st


def init_session_state():
    """Initialize default session_state keys used across the app."""
    defaults = {
        "comparison_result": None,
        "threshold": 0.5,
        "n_components": 50,
        "artifacts_loaded": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
