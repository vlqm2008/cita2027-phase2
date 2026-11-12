"""
Benchmark Script for Jump-Diffusion Adaptation (JumpDiff-H3, Theorem 7).
Generates paper_figures/fig27_jump_diffusion_bipower_variation.png and table_jump_diffusion.csv.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.jump_diffusion_adaptation import (
    JumpDiffusionCrossHorizonAdapter,
)
from cita2027.adaptation.metrics import crps_empirical_samples


def run_jump_diffusion_benchmarks():
    print("[JumpDiff-H3] Running jump-diffusion bipower variation benchmarks...")
    os.makedirs("paper_figures", exist_ok=True)
    horizons = [1, 2, 4, 8, 16]
    T = 120
    np.random.seed(1337)

    # 1. Synthesize series with continuous diffusion and discrete Poisson jump shocks
    continuous_vol = 0.4
    rv_history = []
    bv_history = []
    jump_flags = []
    residuals_raw = []

    # Pre-generate jump locations (flash-crash / grid trip events)
    jump_timesteps = {25, 45, 75, 100}

    for t in range(T):
        if t in jump_timesteps:
            shock = np.random.choice([-1.0, 1.0]) * np.random.uniform(3.5, 5.5)
            jump_flags.append(True)
        else:
            shock = np.random.normal(0, continuous_vol)
            jump_flags.append(False)
        residuals_raw.append(shock)

    # Compare Standard H3 vs JumpDiff-H3
    standard_h3 = HypothesisH3_OverlapAwareCorrector(horizons=horizons, latent_dim=2, lr_shared=0.10)
    jump_h3 = JumpDiffusionCrossHorizonAdapter(
        horizons=horizons,
        latent_dim=2,
        lr_shared=0.10,
        jump_damping_kappa=2.5,
        jump_cushion_beta=1.8,
    )

    std_norm_history = []
    jump_norm_history = []
    scale_cushion_history = []
    jump_intensity_history = []

    crps_std_regular = []
    crps_std_jumps = []
    crps_jump_regular = []
    crps_jump_jumps = []

    for t in range(T):
        # Create feedback items for maturing forecasts
        items = []
        for h in horizons:
            if t >= h:
                y_obs = float(residuals_raw[t])
                base_samples = np.random.normal(0.0, 0.5, size=30)
                items.append(
                    ForecastFeedbackItem(
                        origin_t=t - h,
                        horizon=h,
                        target_t=t,
                        y_true=y_obs,
                        base_samples=base_samples,
                    )
                )

        if items:
            # Update Standard H3
            standard_h3.update_with_feedback(items, current_t=t)
            # Update JumpDiff-H3
            res_j = jump_h3.update_with_feedback(items, current_t=t)

            std_norm_history.append(float(np.linalg.norm(standard_h3.shared_state)))
            jump_norm_history.append(float(np.linalg.norm(jump_h3.base_corrector.shared_state)))
            jump_intensity_history.append(res_j["mean_jump_intensity"])

            # Evaluate CRPS at h=4
            loc_s, scale_s = standard_h3.get_corrections(4)
            loc_j, scale_j = jump_h3.get_corrections(4)
            scale_cushion_history.append(scale_j)

            base_eval = np.random.normal(0.0, 0.5, size=40)
            pred_std = base_eval + loc_s + scale_s * base_eval
            pred_jump = base_eval + loc_j + scale_j * base_eval

            crps_s = crps_empirical_samples(pred_std, residuals_raw[t])
            crps_j = crps_empirical_samples(pred_jump, residuals_raw[t])

            if t in jump_timesteps:
                crps_std_jumps.append(crps_s)
                crps_jump_jumps.append(crps_j)
            else:
                crps_std_regular.append(crps_s)
                crps_jump_regular.append(crps_j)
        else:
            std_norm_history.append(0.0)
            jump_norm_history.append(0.0)
            jump_intensity_history.append(0.0)
            scale_cushion_history.append(1.0)

    # Save summary table
    table_data = [
        {
            "regime": "Normal Continuous Diffusion",
            "standard_h3_crps": float(np.mean(crps_std_regular)),
            "jumpdiff_h3_crps": float(np.mean(crps_jump_regular)),
            "relative_improvement": float(
                (np.mean(crps_std_regular) - np.mean(crps_jump_regular)) / np.mean(crps_std_regular) * 100
            ),
        },
        {
            "regime": "Poisson Jump Discontinuity",
            "standard_h3_crps": float(np.mean(crps_std_jumps)),
            "jumpdiff_h3_crps": float(np.mean(crps_jump_jumps)),
            "relative_improvement": float(
                (np.mean(crps_std_jumps) - np.mean(crps_jump_jumps)) / np.mean(crps_std_jumps) * 100
            ),
        },
    ]
    df = pd.DataFrame(table_data)
    df.to_csv("table_jump_diffusion.csv", index=False)
    print("Saved table_jump_diffusion.csv")

    # Plot Figure 27
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Residual Series & Detected Jump Shocks
    ax = axes[0, 0]
    ax.plot(residuals_raw, color="#4a4a4a", linewidth=1.2, label="Residual Innovation e_t")
    for j_t in jump_timesteps:
        ax.axvline(j_t, color="crimson", linestyle="--", alpha=0.8, linewidth=1.5)
        ax.plot(j_t, residuals_raw[j_t], "r*", markersize=12)
    ax.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax.set_title("(a) Residual Innovation Series with Poisson Jumps", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time Step t", fontsize=10)
    ax.set_ylabel("Standardized Residual", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(["Continuous Innovations", "Jump Arrival (t in {25,45,75,100})"], fontsize=9)

    # (b) Manifold Stability: Standard H3 vs JumpDiff-H3
    ax = axes[0, 1]
    ax.plot(std_norm_history, color="#d62728", linestyle="--", linewidth=1.5, label="Standard H3 (Overshoots)")
    ax.plot(jump_norm_history, color="#1f77b4", linewidth=2.0, label="JumpDiff-H3 (Shielded Stability)")
    for j_t in jump_timesteps:
        ax.axvline(j_t, color="crimson", linestyle=":", alpha=0.4)
    ax.set_title("(b) Representation Manifold Norm ||z_t||_2 under Jumps", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time Step t", fontsize=10)
    ax.set_ylabel("State Norm ||z_t||_2", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # (c) Dynamic Scale Cushion s_{t,h} Response
    ax = axes[1, 0]
    ax.plot(scale_cushion_history, color="#2ca02c", linewidth=2.0, label="Effective Scale Multiplier s_{t, h=4}")
    ax.plot(jump_intensity_history, color="#ff7f0e", linestyle="-.", linewidth=1.5, label="Jump Intensity Ratio J_t")
    ax.set_title("(c) Bipower Jump Cushion & Dispersion Expansion", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time Step t", fontsize=10)
    ax.set_ylabel("Multiplier / Intensity", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # (d) CRPS Breakdown: Regular vs Jump Regimes
    ax = axes[1, 1]
    categories = ["Normal Diffusion", "Poisson Jump Shock"]
    std_means = [np.mean(crps_std_regular), np.mean(crps_std_jumps)]
    jump_means = [np.mean(crps_jump_regular), np.mean(crps_jump_jumps)]
    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width / 2, std_means, width, label="Standard H3", color="#ff7f0e", alpha=0.85, edgecolor="black")
    ax.bar(x + width / 2, jump_means, width, label="JumpDiff-H3", color="#1f77b4", alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_title("(d) CRPS Distributional Error by Volatility Regime", fontsize=11, fontweight="bold")
    ax.set_ylabel("CRPS (Lower is Better)", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=10)

    # Annotate percentage drop on jump regime
    pct_drop = float((std_means[1] - jump_means[1]) / std_means[1] * 100)
    ax.annotate(
        f"-{pct_drop:.1f}% CRPS Reduction\nunder Jump Shocks",
        xy=(1 + width / 2, jump_means[1]),
        xytext=(0.7, jump_means[1] + 0.3),
        arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
        fontsize=9,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig("paper_figures/fig27_jump_diffusion_bipower_variation.png", dpi=300)
    plt.close()
    print("Saved paper_figures/fig27_jump_diffusion_bipower_variation.png")


if __name__ == "__main__":
    run_jump_diffusion_benchmarks()
