"""Streamlit entrypoint for the face recognition (PCA/SVD) app."""

import os
import sys

# Make the project root importable so `core` and `data` packages resolve
# regardless of where `streamlit run` is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
from PIL import Image

from app import state
from app.components import calibration_view, eigenspace_explorer, result_view, upload_widget

st.set_page_config(
    page_title="Face Recognition (PCA/SVD)",
    page_icon="🔍",
    layout="wide",
)
state.init_session_state()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Pengaturan")

    st.session_state["threshold"] = st.slider(
        "Threshold keputusan",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["threshold"],
        step=0.01,
        help="Confidence di atas threshold → SAMA, di bawah → BEDA.",
    )

    st.session_state["n_components"] = st.slider(
        "Jumlah Komponen PCA (k)",
        min_value=1,
        max_value=150,
        value=st.session_state.get("n_components", 150),
        step=1,
        help="Semakin kecil k, semakin banyak detail yang diabaikan. (Maksimal 150 sesuai training LFW)",
    )

    st.divider()
    st.caption(
        "**Face Recognition (PCA/SVD)**\n\n"
        "Membandingkan dua wajah tanpa deep learning — murni Eigenfaces, "
        "preprocessing, dan multi-metric matching."
    )

# --- Main content ---
st.title("🔍 Face Recognition dengan PCA/SVD")
st.caption(
    "Membandingkan dua wajah tanpa deep learning — murni Eigenfaces, "
    "preprocessing, dan multi-metric matching."
)

tab_compare, tab_eigenspace, tab_calibration = st.tabs(
    ["🔄 Bandingkan wajah", "🧠 Eksplorasi eigenspace", "📊 Kalibrasi & validasi"]
)

with tab_compare:
    image_a, image_b = upload_widget.render()

    if image_a is not None and image_b is not None:
        if st.button("🔍 Bandingkan", type="primary", use_container_width=True):
            with st.spinner("Memproses wajah..."):
                try:
                    # Convert uploaded files to grayscale numpy arrays
                    img_a = np.array(
                        Image.open(image_a).convert("L"), dtype=np.uint8
                    )
                    img_b = np.array(
                        Image.open(image_b).convert("L"), dtype=np.uint8
                    )

                    from core.pipeline import compare

                    result = compare(
                        img_a,
                        img_b,
                        config={
                            "threshold": st.session_state["threshold"],
                            "n_components": st.session_state["n_components"],
                        },
                    )
                    st.session_state["comparison_result"] = result

                except FileNotFoundError as e:
                    st.error(
                        f"⚠️ Artifacts belum dibangun. Jalankan:\n\n"
                        f"```\npython scripts/build_eigenspace.py\n```\n\n"
                        f"Detail: {e}"
                    )
                except ValueError as e:
                    st.warning(f"⚠️ {e}")
                except Exception as e:
                    st.error(f"❌ Terjadi error: {e}")

    result_view.render(st.session_state["comparison_result"])

with tab_eigenspace:
    eigenspace_explorer.render()

with tab_calibration:
    calibration_view.render()
