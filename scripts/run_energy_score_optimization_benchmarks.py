"""
Benchmark: Direct Multivariate Energy Score Online Optimization (ES-H3).
Compares direct Energy Score subgradient optimization against standard squared-error H3.
Outputs:
- table_energy_score_optimization.csv
- paper_figures/fig22_energy_score_direct_optimization.png (300 DPI)
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

from cita2027.adaptation.energy_score_optimization import HypothesisESH3_DirectOptimizer
from cita2027.adaptation.feedback_queue import CausalFeedbackQueue, ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.distribution_metrics import energy_score, variogram_score
from cita2027.adaptation.metrics import crps_empirical_samples


def run_benchmark():
    horizons = [1, 2, 4, 8, 16]
    length = 320
    rng = np.random.default_rng(101)

    series = np.cumsum(rng.normal(0, 0.4, size=length))
    # Multivariate correlated shift at t=130
    shift = np.zeros(length)
    shift[130:240] = 3.2
    vol_scale = np.ones(length)
    vol_scale[130:240] = 1.8

    models = {
        "Frozen Zero-Shot (B0)": None,
        "Unified Adapter (H3)": HypothesisH3_OverlapAwareCorrector(horizons, latent_dim=2),
        "Direct Energy Score (ES-H3)": HypothesisESH3_DirectOptimizer(horizons, latent_dim=2),
    }

    es_history = {m: [] for m in models}
    crps_history = {m: [] for m in models}
    vario_history = {m: [] for m in models}

    queue = CausalFeedbackQueue()

    for t in range(length - max(horizons)):
        true_traj = np.array([series[t + h] + shift[t + h] for h in horizons], dtype=np.float64)
        base_ensemble_dict = {}

        for h in horizons:
            loc = series[t]
            disp = math.sqrt(1.0 + 0.1 * h) * vol_scale[t]
            samples = rng.normal(loc, disp, size=100)
            base_ensemble_dict[h] = samples
            queue.enqueue(
                ForecastFeedbackItem(origin_t=t, horizon=h, target_t=t + h, y_true=true_traj[horizons.index(h)], base_samples=samples)
            )

        # Build trajectory matrix for each model
        for m_name, model in models.items():
            traj_samples = np.zeros((100, len(horizons)), dtype=np.float64)
            for j, h in enumerate(horizons):
                base_s = base_ensemble_dict[h]
                loc = series[t]
                if model is None:
                    adapted_s = base_s
                elif m_name == "Unified Adapter (H3)":
                    d_mu, s_mult = model.get_corrections(h)
                    adapted_s = (loc + d_mu) + s_mult * (base_s - loc)
                else:  # ES-H3
                    adapted_s = model.adapt_samples(base_s, loc, h)

                traj_samples[:, j] = adapted_s
                # Marginal CRPS
                c = crps_empirical_samples(adapted_s, true_traj[j])
                crps_history[m_name].append(c)

            # Multivariate Energy Score
            es_val = energy_score(true_traj, traj_samples)
            es_history[m_name].append(es_val)

            # Variogram score order 0.5
            vario_val = variogram_score(true_traj, traj_samples, p=0.5)
            vario_history[m_name].append(vario_val)

        matured = queue.step_and_release(t + 1)
        if matured:
            for m_name, model in models.items():
                if model is not None:
                    model.update_with_feedback(matured, current_t=t + 1)

    # Summary table
    table_rows = []
    for m_name in models:
        table_rows.append({
            "Model": m_name,
            "Energy_Score": round(float(np.mean(es_history[m_name])), 4),
            "Variogram_Score_p05": round(float(np.mean(vario_history[m_name])), 4),
            "Mean_CRPS": round(float(np.mean(crps_history[m_name])), 4),
        })

    df = pd.DataFrame(table_rows)
    csv_path = ROOT_DIR / "table_energy_score_optimization.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Exported Energy Score Table to {csv_path}")

    # Plot Figure 22
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Panel A: Rolling Multivariate Energy Score Trajectory
    window = 35
    for m_name, color, style in [
        ("Frozen Zero-Shot (B0)", "#94a3b8", "--"),
        ("Unified Adapter (H3)", "#3b82f6", "-."),
        ("Direct Energy Score (ES-H3)", "#10b981", "-"),
    ]:
        series_roll = pd.Series(es_history[m_name]).rolling(window, min_periods=10).mean()
        axes[0].plot(series_roll, label=m_name, color=color, linestyle=style, linewidth=2.0)

    axes[0].axvspan(130, 240, color="#fef3c7", alpha=0.5, label="Multivariate Structural Shift")
    axes[0].set_xlabel("Prequential Time Step (t)", fontweight="bold")
    axes[0].set_ylabel("Rolling Energy Score (Lower is Better)", fontweight="bold")
    axes[0].set_title("(A) Rolling Multivariate Energy Score Tracking", fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=True)

    # Panel B: Comparative Metric Bar Chart
    metrics = ["Energy Score", "Variogram (p=0.5)", "Mean CRPS"]
    x = np.arange(len(metrics))
    width = 0.26
    colors = ["#94a3b8", "#3b82f6", "#10b981"]

    for i, m_name in enumerate(models):
        vals = [
            df.loc[df["Model"] == m_name, "Energy_Score"].values[0],
            df.loc[df["Model"] == m_name, "Variogram_Score_p05"].values[0],
            df.loc[df["Model"] == m_name, "Mean_CRPS"].values[0],
        ]
        axes[1].bar(x + (i - 1) * width, vals, width, label=m_name, color=colors[i])

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics, fontweight="bold")
    axes[1].set_ylabel("Metric Value (Lower is Better)", fontweight="bold")
    axes[1].set_title("(B) Trajectory Distribution Scores Comparison", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    fig_path = ROOT_DIR / "paper_figures" / "fig22_energy_score_direct_optimization.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved Figure 22 to {fig_path}")


if __name__ == "__main__":
    run_benchmark()
