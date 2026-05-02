"""Streamlit session-state helpers."""

import streamlit as st

DEFAULT_SESSION_STATE = {
    "analysis_results": None,
    "claims": [],
    "selected_claims": [],
    "evidence": [],
    "verdicts": [],
    "baseline_result": None,
    "stylistic_features": None,
    "use_trust_agents": True,
    "pipeline": "trust",
}


def init_session_state() -> None:
    """Initialize Streamlit session state defaults."""
    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value
