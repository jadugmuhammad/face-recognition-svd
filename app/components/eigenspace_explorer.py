"""Visualizes the trained eigenspace: mean face, top eigenfaces, scree plot."""

import streamlit as st


def render():
    """Render the eigenspace exploration tab."""
    st.info(
        "Eigenspace belum dilatih. Jalankan `scripts/build_eigenspace.py` "
        "lalu modul ini akan menampilkan mean face, top eigenfaces, dan scree plot."
    )
    # TODO: load artifacts/eigenspace.npz dan visualisasikan
