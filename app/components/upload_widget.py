"""Widget for uploading the two face images to compare."""

import streamlit as st


def render():
    """Render the two-image upload widget.

    Returns:
        tuple: the two uploaded files (UploadedFile or None each).
    """
    col1, col2 = st.columns(2)
    with col1:
        image_a = st.file_uploader(
            "Gambar 1", type=["jpg", "jpeg", "png", "pgm"], key="image_a"
        )
        if image_a is not None:
            st.image(image_a, use_container_width=True)
    with col2:
        image_b = st.file_uploader(
            "Gambar 2", type=["jpg", "jpeg", "png", "pgm"], key="image_b"
        )
        if image_b is not None:
            st.image(image_b, use_container_width=True)
    return image_a, image_b
