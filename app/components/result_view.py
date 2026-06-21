"""Renders comparison results: metric breakdown, decision, and reconstructions."""

import numpy as np
import streamlit as st


def render(result=None):
    """Render comparison results.

    Args:
        result: A core.pipeline.FaceComparisonResult, or None if no
            comparison has been run yet.
    """
    if result is None:
        st.info(
            "Upload dua gambar lalu klik **Bandingkan** untuk melihat "
            "hasil di sini."
        )
        return

    # --- Decision banner ---
    if result.is_same:
        st.success(
            f"✅ **SAMA** — Confidence: **{result.confidence:.4f}** "
            f"(threshold: {result.threshold:.4f})"
        )
    else:
        st.error(
            f"❌ **BEDA** — Confidence: **{result.confidence:.4f}** "
            f"(threshold: {result.threshold:.4f})"
        )

    # --- Confidence bar ---
    st.progress(
        min(max(result.confidence, 0.0), 1.0),
        text=f"Confidence: {result.confidence:.4f}",
    )

    # --- Metric breakdown table ---
    st.subheader("Breakdown per metrik")

    metric_data = []
    for metric in ["euclidean", "cosine", "mahalanobis"]:
        raw = result.metric_scores.get(metric, 0)
        conf = result.metric_confidences.get(metric, 0)
        metric_data.append({
            "Metrik": metric.capitalize(),
            "Jarak (raw)": f"{raw:.6f}",
            "Confidence": f"{conf:.4f}",
        })

    st.table(metric_data)

    # --- Reconstructions ---
    if result.reconstruction_a is not None and result.reconstruction_b is not None:
        st.subheader("Rekonstruksi PCA")
        st.caption(
            "Gambar yang direkonstruksi dari koefisien eigenspace — "
            "menunjukkan apa yang 'dilihat' oleh PCA."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Gambar A (rekonstruksi)**")
            # Normalize to 0-255 for display
            img_a = _normalize_for_display(result.reconstruction_a)
            st.image(img_a, use_container_width=True)
        with col2:
            st.write("**Gambar B (rekonstruksi)**")
            img_b = _normalize_for_display(result.reconstruction_b)
            st.image(img_b, use_container_width=True)


def _normalize_for_display(image):
    """Normalize a float image to uint8 for Streamlit display."""
    img = np.array(image, dtype=np.float64)
    # Clip and scale to 0-255
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max() * 255.0
    return img.astype(np.uint8)
