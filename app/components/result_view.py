"""Renders comparison results: metric breakdown, decision, and reconstructions."""

import streamlit as st


def render(result=None):
    """Render comparison results.

    Args:
        result: A core.pipeline.FaceComparisonResult, or None if no
            comparison has been run yet.
    """
    if result is None:
        st.info("Upload dua gambar lalu klik 'Bandingkan' untuk melihat hasil di sini.")
        return
    # TODO: tampilkan tabel breakdown 3 metrik, confidence score, keputusan,
    # dan gambar rekonstruksi PCA setelah core.pipeline siap.
    st.write(result)
