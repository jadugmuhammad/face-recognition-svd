"""Streamlit entrypoint for the face recognition (PCA/SVD) app."""

import os
import sys

# Make the project root importable so `core` and `data` packages resolve
# regardless of where `streamlit run` is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app import state
from app.components import calibration_view, eigenspace_explorer, result_view, upload_widget

st.set_page_config(page_title="Face Recognition (PCA/SVD)", layout="wide")
state.init_session_state()

st.title("Face recognition dengan PCA/SVD")
st.caption(
    "Membandingkan dua wajah tanpa deep learning -- murni Eigenfaces, "
    "preprocessing, dan multi-metric matching."
)

tab_compare, tab_eigenspace, tab_calibration = st.tabs(
    ["Bandingkan wajah", "Eksplorasi eigenspace", "Kalibrasi & validasi"]
)

with tab_compare:
    image_a, image_b = upload_widget.render()

    threshold = st.slider(
        "Threshold", min_value=0.0, max_value=1.0,
        value=st.session_state["threshold"], step=0.01,
    )
    st.session_state["threshold"] = threshold

    if image_a is not None and image_b is not None:
        if st.button("Bandingkan", type="primary"):
            st.warning(
                "core.pipeline.compare belum diimplementasikan -- ini baru "
                "placeholder dari tahap setup struktur repo."
            )
            # TODO: panggil core.pipeline.compare(image_a, image_b, config) di sini
            # dan simpan hasilnya ke st.session_state["comparison_result"]

    result_view.render(st.session_state["comparison_result"])

with tab_eigenspace:
    eigenspace_explorer.render()

with tab_calibration:
    calibration_view.render()
