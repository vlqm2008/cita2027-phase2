"""
Benchmark Script for Streaming Change-Point Martingale (CPM-H3, Theorem 9).
Generates paper_figures/fig31_streaming_changepoint_martingale.png and table_cpm_benchmarks.csv.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cita2027.adaptation.changepoint_martingale import (
    ChangePointMartingaleAdapter,
    StreamingChangePointMartingale,
)
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector


def run_cpm_benchmarks():
    print("[CPM-H3] Running streaming change-point martingale benchmarks...")
    os.makedirs("paper_figures", exist_ok=True)
    horizons = [1, 2, 4, 8, 16]
    T = 100
    np.random.seed(42)

    # Abrupt regime shift at t=45: bias jumps from +0.5 to -2.0
    true_bias = np.array([0.5 if t < 45 else -2.0 for t in range(T)])

    fixed_adapter = HypothesisH3_OverlapAwareCorrector(horizons=horizons, latent_dim=2, lr_shared=0.06)
    cpm_adapter = ChangePointMartingaleAdapter(
        horizons=horizons,
        latent_dim=2,
        base_lr=0.06,
        burst_multiplier=3.5,
        cpm_threshold=18.0,
    )

    martingale_vals = []
    lr_vals = []
    fixed_errs = []
    cpm_errs = []
    detected_timesteps = []

    for t in range(T):
        items = []
        for h in horizons:
            if t >= h:
                y_obs = float(true_bias[t] + 0.1 * np.log2(h) + np.random.normal(0, 0.25))
                items.append(
                    ForecastFeedbackItem(
                        origin_t=t - h,
                        horizon=h,
                        target_t=t,
                        y_true=y_obs,
                        base_samples=np.random.normal(0, 0.5, size=20),
                    )
                )

        # Update fixed
        fixed_adapter.update_with_feedback(items, current_t=t)
        # Update CPM
        res_c = cpm_adapter.update_with_feedback(items, current_t=t)

        martingale_vals.append(res_c["martingale_value"])
        lr_vals.append(res_c["effective_lr"])
        if res_c["change_point_detected"] > 0:
            detected_timesteps.append(t)

        loc_fixed, _ = fixed_adapter.get_corrections(4)
        loc_cpm, _ = cpm_adapter.get_corrections(4)

        fixed_errs.append(abs(loc_fixed - true_bias[t]))
        cpm_errs.append(abs(loc_cpm - true_bias[t]))

    # Save summary table
    df = pd.DataFrame({
        "time_step": range(T),
        "true_bias": true_bias,
        "martingale_value": martingale_vals,
        "effective_lr": lr_vals,
        "fixed_abs_error": fixed_errs,
        "cpm_abs_error": cpm_errs,
    })
    df.to_csv("table_cpm_benchmarks.csv", index=False)
    print("Saved table_cpm_benchmarks.csv")

    # Plot Figure 31
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) True Shift vs Estimated Corrections
    ax = axes[0, 0]
    ax.plot(true_bias, "k--", linewidth=2.0, label="True Environmental Bias")
    ax.plot([b - e for b, e in zip(true_bias, fixed_errs)], color="#ff7f0e", linestyle=":", linewidth=1.5, label="Fixed LR (Slow Drift)")
    ax.plot([b - e for b, e in zip(true_bias, cpm_errs)], color="#1f77b4", linewidth=2.0, label="CPM-H3 (Rapid Recovery)")
    ax.axvline(45, color="red", linestyle="--", alpha=0.7, label="Abrupt Break (t=45)")
    ax.set_title("(a) Abrupt Structural Shift Tracking (Horizon h=4)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time Step t", fontsize=10)
    ax.set_ylabel("Bias Correction", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # (b) Shiryaev-Roberts Change-Point Martingale
    ax = axes[0, 1]
    ax.plot(martingale_vals, color="#d62728", linewidth=2.0, label="Martingale Lambda_t")
    ax.axhline(18.0, color="black", linestyle="--", linewidth=1.5, label="Alarm Threshold B=18.0")
    for d_t in detected_timesteps:
        ax.plot(d_t, 18.0, "r*", markersize=14, label="Alarm Declared" if d_t == detected_timesteps[0] else "")
    ax.set_title("(b) Sequential Likelihood Ratio Martingale Lambda_t", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time Step t", fontsize=10)
    ax.set_ylabel("Martingale Value", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # (c) Dynamic Burst Learning Rate Schedule
    ax = axes[1, 0]
    ax.plot(lr_vals, color="#2ca02c", linewidth=2.0, label="Adaptive LR eta(t)")
    ax.axhline(0.06, color="black", linestyle=":", alpha=0.7, label="Base LR eta_0=0.06")
    ax.axvline(45, color="red", linestyle="--", alpha=0.7)
    ax.set_title("(c) Dynamic Burst Learning Rate Scheduling", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time Step t", fontsize=10)
    ax.set_ylabel("Learning Rate eta", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # (d) Post-Break Recovery Trajectory
    ax = axes[1, 1]
    post_break_steps = range(45, 75)
    ax.plot(list(post_break_steps), [fixed_errs[t] for t in post_break_steps], "o-", color="#ff7f0e", linewidth=1.8, label="Fixed LR Error")
    ax.plot(list(post_break_steps), [cpm_errs[t] for t in post_break_steps], "s-", color="#1f77b4", linewidth=2.2, label="CPM-H3 Error")
    ax.set_title("(d) Post-Break Error Recovery (Steps 45-75)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time Step t", fontsize=10)
    ax.set_ylabel("Absolute Tracking Error", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("paper_figures/fig31_streaming_changepoint_martingale.png", dpi=300)
    plt.close()
    print("Saved paper_figures/fig31_streaming_changepoint_martingale.png")


if __name__ == "__main__":
    run_cpm_benchmarks()
