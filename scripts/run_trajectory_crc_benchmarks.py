"""
Benchmark: Asynchronous Trajectory Conformal Risk Control (AT-CRC).
Evaluates joint multi-horizon coverage guarantees under non-stationary shifts.
Outputs:
- table_trajectory_crc.csv
- paper_figures/fig19_trajectory_conformal_risk_control.png (300 DPI)
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from cita2027.adaptation.feedback_queue import CausalFeedbackQueue, ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.trajectory_conformal_risk_control import AsynchronousTrajectoryCRC


def run_benchmark():
    horizons = [1, 2, 4, 8, 16]
    num_horizons = len(horizons)
    length = 350
    rng = np.random.default_rng(42)

    # Simulate non-stationary trajectory with abrupt break at t=150
    series = np.cumsum(rng.normal(0, 0.4, size=length))
    shift = np.zeros(length)
    shift[150:260] = 3.5  # Structural break
    vol_mult = np.ones(length)
    vol_mult[150:260] = 2.0

    crc_target_alpha = 0.10  # 90% joint trajectory coverage
    crc = AsynchronousTrajectoryCRC(
        horizons=horizons,
        alpha_joint=crc_target_alpha,
        learning_rate=0.20,
        initial_quantile=2.6,
        max_quantile=8.0,
    )
    adapter = HypothesisH3_OverlapAwareCorrector(horizons=horizons, latent_dim=2)
    queue = CausalFeedbackQueue()

    # Track trajectory outcomes
    trajectory_covered_standard = []
    trajectory_covered_crc = []
    joint_quantile_history = []
    band_widths_history = {h: [] for h in horizons}

    for t in range(length - max(horizons)):
        crc.register_forecast(origin_t=t)
        base_loc = {}
        base_disp = {}
        adapted_loc = {}
        adapted_scale = {}
        true_vals = {}
        base_samples_dict = {}

        for h in horizons:
            true_vals[h] = series[t + h] + shift[t + h]
            loc = series[t]
            disp = math.sqrt(1.0 + 0.1 * h) * vol_mult[t]
            samples = rng.normal(loc, disp, size=100)

            base_loc[h] = loc
            base_disp[h] = disp
            base_samples_dict[h] = samples

            # Adapter corrections
            d_mu, s_mult = adapter.get_corrections(h)
            adapted_loc[h] = loc + d_mu
            adapted_scale[h] = s_mult
            adapted_samples = (loc + d_mu) + s_mult * (samples - loc)

            # Enqueue feedback
            queue.enqueue(
                ForecastFeedbackItem(
                    origin_t=t,
                    horizon=h,
                    target_t=t + h,
                    y_true=true_vals[h],
                    base_samples=samples,
                    adapted_samples=adapted_samples,
                )
            )

        # 1. Standard Independent Conformal / Gaussian 90% (q = 1.645)
        covered_std_all = True
        for h in horizons:
            std_half_w = 1.645 * (base_disp[h] * adapted_scale[h])
            if not (adapted_loc[h] - std_half_w <= true_vals[h] <= adapted_loc[h] + std_half_w):
                covered_std_all = False
        trajectory_covered_standard.append(1.0 if covered_std_all else 0.0)

        # 2. Trajectory Conformal Risk Control Bands
        crc_bands = crc.get_prediction_bands(
            base_locations=base_loc,
            base_dispersions=base_disp,
            adapted_locations=adapted_loc,
            adapted_scales=adapted_scale,
        )

        covered_crc_all = True
        for h in horizons:
            l, u, w = crc_bands[h]
            band_widths_history[h].append(w)
            if not (l <= true_vals[h] <= u):
                covered_crc_all = False
        trajectory_covered_crc.append(1.0 if covered_crc_all else 0.0)

        joint_quantile_history.append(crc.q_joint)

        # Matured feedback release
        matured = queue.step_and_release(t + 1)
        if matured:
            adapter.update_with_feedback(matured, current_t=t + 1)
            crc.update_with_feedback(matured, current_t=t + 1)

    # Compute rolling coverage (window=50)
    window = 40
    rolling_cov_std = pd.Series(trajectory_covered_standard).rolling(window, min_periods=10).mean()
    rolling_cov_crc = pd.Series(trajectory_covered_crc).rolling(window, min_periods=10).mean()

    # Summary table
    table_data = [
        {
            "Method": "Independent Pointwise 90%",
            "Joint_Trajectory_Coverage": round(float(np.mean(trajectory_covered_standard)), 4),
            "Mean_Band_Width_h1": round(float(np.mean(band_widths_history[1])) * (1.645 / np.mean(joint_quantile_history)), 4),
            "Mean_Band_Width_h16": round(float(np.mean(band_widths_history[16])) * (1.645 / np.mean(joint_quantile_history)), 4),
            "Target_Joint_Coverage": 0.90,
            "Target_Met": float(np.mean(trajectory_covered_standard)) >= 0.90,
        },
        {
            "Method": "Asynchronous Trajectory CRC (AT-CRC)",
            "Joint_Trajectory_Coverage": round(float(np.mean(trajectory_covered_crc)), 4),
            "Mean_Band_Width_h1": round(float(np.mean(band_widths_history[1])), 4),
            "Mean_Band_Width_h16": round(float(np.mean(band_widths_history[16])), 4),
            "Target_Joint_Coverage": 0.90,
            "Target_Met": float(np.mean(trajectory_covered_crc)) >= 0.90,
        },
    ]

    df = pd.DataFrame(table_data)
    csv_path = ROOT_DIR / "table_trajectory_crc.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Exported Trajectory CRC Table to {csv_path}")

    # Plot Figure 19
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=300, sharex=True)

    # Panel A: Rolling Joint Trajectory Coverage
    axes[0].plot(rolling_cov_std, label="Independent Pointwise 90% (Standard Union Bound Failure)", color="#ef4444", linestyle="--", linewidth=1.8)
    axes[0].plot(rolling_cov_crc, label="AT-CRC (Joint Trajectory Calibrator)", color="#10b981", linewidth=2.3)
    axes[0].axhline(0.90, color="#1e293b", linestyle=":", linewidth=1.5, label="Nominal 90% Joint Target")
    axes[0].axvspan(150, 260, color="#fef3c7", alpha=0.5, label="Structural Break Shock (t in [150, 260])")
    axes[0].set_ylabel("Rolling Joint Coverage", fontweight="bold")
    axes[0].set_title("(A) Joint Multi-Horizon Trajectory Coverage Tracking Under Structural Break", fontweight="bold")
    axes[0].set_ylim(0.40, 1.02)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower left", fontsize=8.5, frameon=True)

    # Panel B: Dynamic Quantile Multiplier q_joint(t)
    axes[1].plot(joint_quantile_history, color="#3b82f6", linewidth=2.0, label="Dynamic Joint Quantile Multiplier q_joint(t)")
    axes[1].axhline(1.645, color="#64748b", linestyle="--", label="Pointwise Gaussian Baseline (1.645)")
    axes[1].axvspan(150, 260, color="#fef3c7", alpha=0.5)
    axes[1].set_xlabel("Prequential Time Step (t)", fontweight="bold")
    axes[1].set_ylabel("Joint Quantile Multiplier", fontweight="bold")
    axes[1].set_title("(B) Causal Quantile Adaptation Dynamics", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper left", fontsize=8.5, frameon=True)

    plt.tight_layout()
    fig_path = ROOT_DIR / "paper_figures" / "fig19_trajectory_conformal_risk_control.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved Figure 19 to {fig_path}")


if __name__ == "__main__":
    run_benchmark()
