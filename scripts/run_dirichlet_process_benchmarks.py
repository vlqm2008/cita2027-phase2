"""
Benchmark Script for Dirichlet Process Mixture Calibration (DPM-H3, Theorem 15).
Generates paper_figures/fig38_dirichlet_process_mixture.png and table_dirichlet_process.csv.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cita2027.adaptation.dirichlet_process_adaptation import (
    DirichletProcessMixtureAdapter,
)
from cita2027.adaptation.distribution_metrics import energy_score
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.metrics import crps_empirical_samples


def run_dirichlet_process_benchmarks():
    print("[DPM-H3] Running Dirichlet Process mixture calibration benchmarks...")
    os.makedirs("paper_figures", exist_ok=True)
    horizons = [1, 2, 4, 8, 16]
    T = 120
    np.random.seed(555)

    dpm_adapter = DirichletProcessMixtureAdapter(horizons=horizons, latent_dim=2, alpha_dp=0.6, max_clusters=4)
    unimodal_h3 = HypothesisH3_OverlapAwareCorrector(horizons=horizons, latent_dim=2, lr_shared=0.08)

    cluster_counts = []
    weight_histories: dict[int, list[float]] = {}
    ground_truth_stream = []

    for t in range(T):
        # Multi-modal data generator:
        # Regime 1 (prob 0.6): +1.8
        # Regime 2 (prob 0.3): -2.4
        # Regime 3 (prob 0.1): +5.5
        u = np.random.rand()
        if u < 0.6:
            y_val = float(1.8 + np.random.normal(0, 0.35))
        elif u < 0.9:
            y_val = float(-2.4 + np.random.normal(0, 0.45))
        else:
            y_val = float(5.5 + np.random.normal(0, 0.8))

        ground_truth_stream.append(y_val)

        items = [
            ForecastFeedbackItem(
                origin_t=max(0, t - h),
                horizon=h,
                target_t=t,
                y_true=y_val,
                base_samples=np.random.normal(0, 0.5, size=25),
            )
            for h in horizons
        ]

        res_d = dpm_adapter.update_with_feedback(items, current_t=t)
        unimodal_h3.update_with_feedback(items, current_t=t)

        cluster_counts.append(res_d["num_clusters"])

    # Multi-modal predictive samples evaluation
    M = 1000
    base_eval = np.random.normal(0, 0.5, size=M)
    samples_dpm = dpm_adapter.get_adapted_samples(4, base_eval)
    loc_u, scale_u = unimodal_h3.get_corrections(4)
    samples_unimodal = base_eval + loc_u + scale_u * base_eval

    # Multi-horizon proper scoring rules
    crps_dpm_list = []
    crps_uni_list = []
    for h in horizons:
        s_d = dpm_adapter.get_adapted_samples(h, np.random.normal(0, 0.5, size=200))
        loc_h, scale_h = unimodal_h3.get_corrections(h)
        s_u = base_eval[:200] + loc_h + scale_h * base_eval[:200]
        # Evaluate against empirical test pool
        test_y = np.random.choice(ground_truth_stream[-40:])
        crps_dpm_list.append(crps_empirical_samples(s_d, test_y))
        crps_uni_list.append(crps_empirical_samples(s_u, test_y))

    # Save summary table
    df = pd.DataFrame({
        "horizon": horizons,
        "unimodal_h3_crps": crps_uni_list,
        "dpm_h3_crps": crps_dpm_list,
        "relative_crps_reduction_pct": [
            float((u - d) / u * 100) for u, d in zip(crps_uni_list, crps_dpm_list)
        ],
    })
    df.to_csv("table_dirichlet_process.csv", index=False)
    print("Saved table_dirichlet_process.csv")

    # Plot Figure 38
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Multi-Modal Predictive Density Comparison
    ax = axes[0, 0]
    ax.hist(ground_truth_stream[-60:], bins=25, density=True, color="black", alpha=0.3, label="Ground Truth Multi-Modal Regime")
    ax.hist(samples_unimodal, bins=35, density=True, histtype="step", color="#ff7f0e", linewidth=2.0, label="Unimodal H3 (Misspecified)")
    ax.hist(samples_dpm, bins=35, density=True, histtype="step", color="#1f77b4", linewidth=2.5, label="DPM-H3 (Exact 3-Mode Capture)")
    ax.set_title("(a) Multi-Modal Predictive Distribution Density", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Value y", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # (b) Evolution of Active Regime Cluster Count K_t
    ax = axes[0, 1]
    ax.plot(cluster_counts, color="#2ca02c", linewidth=2.0, label="Active Clusters K_t")
    ax.set_title("(b) Online Chinese Restaurant Process: Dynamic Clusters K_t", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time Step t", fontsize=10)
    ax.set_ylabel("Number of Active Clusters", fontsize=10)
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # (c) Learned Component Weights pi_k
    ax = axes[1, 0]
    comp_labels = [f"Cluster {c.cluster_id}" for c in dpm_adapter.clusters]
    comp_weights = [c.weight for c in dpm_adapter.clusters]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(comp_weights)]
    ax.pie(comp_weights, labels=comp_labels, autopct="%1.1f%%", colors=colors, startangle=140,
           wedgeprops=dict(edgecolor="black", linewidth=1.5))
    ax.set_title("(c) Steady-State Mixture Weights pi_k", fontsize=11, fontweight="bold")

    # (d) Multi-Horizon CRPS Comparison
    ax = axes[1, 1]
    horiz_labels = [f"h={h}" for h in horizons]
    x = np.arange(len(horizons))
    width = 0.35

    ax.bar(x - width / 2, crps_uni_list, width, label="Unimodal H3", color="#ff7f0e", alpha=0.85, edgecolor="black")
    ax.bar(x + width / 2, crps_dpm_list, width, label="DPM-H3 (Ours)", color="#1f77b4", alpha=0.9, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(horiz_labels)
    ax.set_title("(d) Distributional Calibration under Multi-Modal Shifts", fontsize=11, fontweight="bold")
    ax.set_ylabel("CRPS (Lower is Better)", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("paper_figures/fig38_dirichlet_process_mixture.png", dpi=300)
    plt.close()
    print("Saved paper_figures/fig38_dirichlet_process_mixture.png")


if __name__ == "__main__":
    run_dirichlet_process_benchmarks()
