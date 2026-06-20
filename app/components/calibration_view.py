"""Visualizes calibration results: ROC curve, EER, score distributions."""

import streamlit as st


def render():
    """Render the calibration/validation tab."""
    st.info(
        "Kalibrasi belum dijalankan. Jalankan `scripts/build_eigenspace.py` "
        "lalu modul ini akan menampilkan ROC curve dan EER dari FG-NET."
    )
    # TODO: load artifacts/calibration.json dan visualisasikan
