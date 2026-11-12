"""
Benchmark: Cross-Horizon Self-Attention Online Adaptation (Attn-H3).
Generates interpretable attention heatmaps between target horizons and matured feedback items.
Outputs:
- table_attention_benchmarks.csv
- paper_figures/fig21_cross_horizon_attention_heatmaps.png (300 DPI)
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

from cita2027.adaptation.attention_cross_horizon import HypothesisAttnH3_Corrector
from cita2027.adaptation.feedback_queue import CausalFeedbackQueue, ForecastFeedbackItem
from cita2027.adaptation.metrics import crps_empirical_samples


def run_benchmark():
    horizons = [1, 2, 4, 8, 16]
    length = 300
    rng = np.random.default_rng(42)

    series = np.cumsum(rng.normal(0, 0.4, size=length))
    shock = np.zeros(length)
    # Structural break at t=120
    shock[120:220] = 3.5

    corrector = HypothesisAttnH3_Corrector(horizons=horizons, embed_dim=8, seed=42)
    queue = CausalFeedbackQueue()

    stationary_attn_matrices = []
    shock_attn_matrices = []
    crps_history = {h: [] for h in horizons}

    for t in range(length - max(horizons)):
        for h in horizons:
            loc = series[t]
            disp = math.sqrt(1.0 + 0.1 * h)
            samples = rng.normal(loc, disp, size=100)
            true_y = series[t + h] + shock[t + h]

            adapted_s = corrector.adapt_samples(samples, loc, h)
            c = crps_empirical_samples(adapted_s, true_y)
            crps_history[h].append(c)

            queue.enqueue(
                ForecastFeedbackItem(origin_t=t, horizon=h, target_t=t + h, y_true=true_y, base_samples=samples)
            )

        matured = queue.step_and_release(t + 1)
        if matured:
            corrector.update_with_feedback(matured, current_t=t + 1)
            mat = corrector.latest_attention_matrix
            # If all horizons matured in this batch
            if mat is not None and mat.shape == (len(horizons), len(horizons)):
                if t < 100:
                    stationary_attn_matrices.append(mat)
                elif 130 <= t <= 190:
                    shock_attn_matrices.append(mat)

    # Compute average attention heatmaps
    mean_attn_stat = np.mean(stationary_attn_matrices, axis=0) if stationary_attn_matrices else np.eye(len(horizons))
    mean_attn_shock = np.mean(shock_attn_matrices, axis=0) if shock_attn_matrices else np.eye(len(horizons))

    # Summary table
    table_rows = []
    for h in horizons:
        table_rows.append({
            "Target_Horizon": f"h={h}",
            "Mean_CRPS": round(float(np.mean(crps_history[h])), 4),
            "Stationary_Self_Attention": round(float(mean_attn_stat[horizons.index(h), horizons.index(h)]), 4),
            "Shock_Cross_Horizon_Attention": round(float(mean_attn_shock[horizons.index(h), 0]), 4),
        })

    df = pd.DataFrame(table_rows)
    csv_path = ROOT_DIR / "table_attention_benchmarks.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Exported Attention Benchmark Table to {csv_path}")

    # Plot Figure 21
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    labels = [f"h={h}" for h in horizons]

    # Panel A: Stationary Attention Heatmap
    im1 = axes[0].imshow(mean_attn_stat, cmap="Blues", vmin=0.0, vmax=0.6)
    axes[0].set_xticks(range(len(horizons)))
    axes[0].set_yticks(range(len(horizons)))
    axes[0].set_xticklabels(labels)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("Matured Feedback Horizon (Key)", fontweight="bold")
    axes[0].set_ylabel("Target Forecast Horizon (Query)", fontweight="bold")
    axes[0].set_title("(A) Stationary Regime: Horizon Attention", fontweight="bold")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    for i in range(len(horizons)):
        for j in range(len(horizons)):
            val = mean_attn_stat[i, j]
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.3 else "black", fontsize=8.5)

    # Panel B: Shock Regime Attention Heatmap
    im2 = axes[1].imshow(mean_attn_shock, cmap="Oranges", vmin=0.0, vmax=0.6)
    axes[1].set_xticks(range(len(horizons)))
    axes[1].set_yticks(range(len(horizons)))
    axes[1].set_xticklabels(labels)
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("Matured Feedback Horizon (Key)", fontweight="bold")
    axes[1].set_ylabel("Target Forecast Horizon (Query)", fontweight="bold")
    axes[1].set_title("(B) Structural Shock Regime: Feedback Routing", fontweight="bold")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    for i in range(len(horizons)):
        for j in range(len(horizons)):
            val = mean_attn_shock[i, j]
            axes[1].text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.3 else "black", fontsize=8.5)

    plt.tight_layout()
    fig_path = ROOT_DIR / "paper_figures" / "fig21_cross_horizon_attention_heatmaps.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved Figure 21 to {fig_path}")


if __name__ == "__main__":
    run_benchmark()
