"""Visualizes the trained eigenspace: mean face, top eigenfaces, scree plot."""

import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

_EIGENSPACE_PATH = os.path.join("artifacts", "eigenspace.npz")


def render():
    """Render the eigenspace exploration tab."""
    if not os.path.isfile(_EIGENSPACE_PATH):
        st.info(
            "Eigenspace belum dilatih. Jalankan "
            "`python scripts/build_eigenspace.py` terlebih dahulu."
        )
        return

    data = np.load(_EIGENSPACE_PATH)
    mean_face = data["mean_face"]
    components = data["components"]
    explained_variance_ratio = data["explained_variance_ratio"]
    image_size = tuple(data["image_size"])  # (width, height)

    img_shape = (image_size[1], image_size[0])  # (height, width)
    n_components = components.shape[0]

    # --- Mean face ---
    st.subheader("Mean face (wajah rata-rata)")
    st.caption(
        "Rata-rata dari semua wajah dalam training set — "
        "ini yang dikurangkan dari setiap wajah sebelum PCA."
    )
    mean_img = _normalize(mean_face.reshape(img_shape))
    st.image(mean_img, width=200)

    # --- Top eigenfaces ---
    st.subheader("Top eigenfaces")
    st.caption(
        "Komponen utama PCA — pola variasi wajah paling dominan. "
        "Komponen pertama menangkap variasi terbesar."
    )

    n_display = st.slider(
        "Jumlah eigenfaces", min_value=5, max_value=min(n_components, 30),
        value=min(10, n_components), key="n_eigenfaces"
    )

    cols_per_row = 5
    for row_start in range(0, n_display, cols_per_row):
        cols = st.columns(cols_per_row)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx >= n_display:
                break
            eigenface = components[idx].reshape(img_shape)
            img = _normalize(eigenface)
            with col:
                st.image(img, caption=f"#{idx + 1}", use_container_width=True)

    # --- Scree plot ---
    st.subheader("Scree plot (explained variance)")
    st.caption(
        "Berapa persen variasi yang ditangkap oleh masing-masing komponen. "
        "Kurva kumulatif menunjukkan total variasi yang di-retain."
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    x = np.arange(1, n_components + 1)
    cumulative = np.cumsum(explained_variance_ratio)

    ax.bar(x, explained_variance_ratio, alpha=0.6, label="Individual")
    ax.plot(x, cumulative, "r-o", markersize=3, label="Kumulatif")
    ax.set_xlabel("Komponen ke-")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(
        f"Scree Plot — {n_components} komponen, "
        f"total {cumulative[-1]:.1%} variasi"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Stats ---
    st.subheader("Statistik")
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah komponen", n_components)
    col2.metric("Total variance", f"{cumulative[-1]:.1%}")
    col3.metric("Dimensi vektor", f"{mean_face.shape[0]:,}")


def _normalize(image):
    """Normalize image to uint8 for display."""
    img = image.astype(np.float64)
    img -= img.min()
    if img.max() > 0:
        img = img / img.max() * 255
    return img.astype(np.uint8)
