"""Visualizes calibration results: ROC curve, EER, score distributions."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

_CALIBRATION_PATH = os.path.join("artifacts", "calibration.json")


def render():
    """Render the calibration/validation tab."""
    if not os.path.isfile(_CALIBRATION_PATH):
        st.info(
            "Kalibrasi belum dijalankan. Jalankan "
            "`python scripts/build_eigenspace.py` terlebih dahulu."
        )
        return

    with open(_CALIBRATION_PATH, "r") as f:
        calibration = json.load(f)

    metrics = ["euclidean", "cosine", "mahalanobis"]

    # --- EER summary ---
    st.subheader("Ringkasan Equal Error Rate (EER)")
    st.caption(
        "EER adalah titik di mana False Positive Rate = False Negative Rate. "
        "Semakin rendah EER, semakin akurat sistem."
    )

    cols = st.columns(len(metrics))
    for i, metric in enumerate(metrics):
        cal = calibration.get(metric, {})
        with cols[i]:
            st.metric(
                metric.capitalize(),
                f"EER: {cal.get('eer', 0):.4f}",
                delta=f"θ = {cal.get('eer_threshold', 0):.4f}",
                delta_color="off",
            )

    ensemble_thresh = calibration.get("ensemble_threshold", 0.5)
    st.info(f"**Ensemble threshold (rata-rata EER):** {ensemble_thresh:.4f}")

    # --- ROC curves ---
    st.subheader("ROC Curves")
    st.caption(
        "Kurva ROC per metrik — semakin mendekati sudut kiri atas, "
        "semakin baik."
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#e94560", "#0f3460", "#533483"]

    for idx, metric in enumerate(metrics):
        cal = calibration.get(metric, {})
        fpr = cal.get("roc_fpr", [])
        tpr = cal.get("roc_tpr", [])
        eer = cal.get("eer", 0)

        ax = axes[idx]
        ax.plot(fpr, tpr, color=colors[idx], linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        ax.plot(
            [eer], [1 - eer], "ro", markersize=8,
            label=f"EER = {eer:.4f}"
        )
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{metric.capitalize()}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Score distributions ---
    st.subheader("Distribusi jarak: Genuine vs Impostor")
    st.caption(
        "Distribusi jarak (raw) untuk pasangan genuine (orang sama) dan "
        "impostor (orang berbeda) dari FG-NET. Overlap yang besar "
        "menandakan metrik kesulitan membedakan."
    )

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))

    for idx, metric in enumerate(metrics):
        cal = calibration.get(metric, {})
        gen_mean = cal.get("genuine_mean", 0)
        gen_std = cal.get("genuine_std", 1)
        imp_mean = cal.get("impostor_mean", 0)
        imp_std = cal.get("impostor_std", 1)

        ax = axes2[idx]

        # Generate approximate distributions for visualization
        x_min = min(gen_mean - 3 * gen_std, imp_mean - 3 * imp_std)
        x_max = max(gen_mean + 3 * gen_std, imp_mean + 3 * imp_std)
        x = np.linspace(x_min, x_max, 200)

        gen_pdf = _gaussian(x, gen_mean, gen_std)
        imp_pdf = _gaussian(x, imp_mean, imp_std)

        ax.fill_between(x, gen_pdf, alpha=0.4, color="#2ecc71", label="Genuine")
        ax.fill_between(x, imp_pdf, alpha=0.4, color="#e74c3c", label="Impostor")
        ax.set_xlabel("Jarak")
        ax.set_ylabel("Densitas")
        ax.set_title(f"{metric.capitalize()}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # --- Raw stats ---
    with st.expander("Detail statistik kalibrasi"):
        for metric in metrics:
            cal = calibration.get(metric, {})
            st.markdown(f"**{metric.capitalize()}**")
            st.json({
                "genuine_mean": cal.get("genuine_mean"),
                "genuine_std": cal.get("genuine_std"),
                "impostor_mean": cal.get("impostor_mean"),
                "impostor_std": cal.get("impostor_std"),
                "eer": cal.get("eer"),
                "eer_threshold": cal.get("eer_threshold"),
            })


def _gaussian(x, mean, std):
    """Simple Gaussian PDF for visualization."""
    if std < 1e-12:
        return np.zeros_like(x)
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mean) / std) ** 2
    )
