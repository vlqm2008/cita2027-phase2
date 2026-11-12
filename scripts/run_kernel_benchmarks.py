"""
Benchmark: Nonlinear Kernelized Cross-Horizon Adaptation (Kernel-H3) vs Linear Manifold.
Evaluates Random Fourier Features under nonlinear cross-horizon term structure twisting.
Outputs:
- table_kernel_benchmarks.csv
- paper_figures/fig18_rff_kernel_vs_linear_manifold.png (300 DPI)
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
from cita2027.adaptation.kernel_adaptation import HypothesisKernelH3_Corrector
from cita2027.adaptation.metrics import crps_empirical_samples


def simulate_nonlinear_term_structure(
    length: int = 350,
    horizons: list[int] = [1, 2, 4, 8, 16],
    seed: int = 42,
) -> dict:
    """
    Generates time-series where cross-horizon shift exhibits nonlinear curvature:
        Delta mu(h) = alpha_t * sin(pi * log2(h) / 3.0) + beta_t * (log2(h) / 4.0)^2
    """
    rng = np.random.default_rng(seed)
    t_idx = np.arange(length)

    # Shift regimes: stationary -> nonlinear curvature twist -> reversal
    alpha = np.zeros(length)
    beta = np.zeros(length)

    # Nonlinear twist between t=100 and t=250
    alpha[100:250] = 1.8 * np.sin(np.linspace(0, math.pi, 150))
    beta[100:250] = 1.2 * np.cos(np.linspace(0, math.pi, 150))

    series = np.cumsum(rng.normal(0, 0.5, size=length))

    return {
        "series": series,
        "alpha": alpha,
        "beta": beta,
        "length": length,
        "horizons": horizons,
    }


def run_benchmark():
    horizons = [1, 2, 4, 8, 16]
    data = simulate_nonlinear_term_structure(length=350, horizons=horizons, seed=42)
    series = data["series"]
    alpha = data["alpha"]
    beta = data["beta"]

    models = {
        "Frozen Zero-Shot (B0)": None,
        "Linear Shared State (H1)": None,
        "Linear Unified (H3)": HypothesisH3_OverlapAwareCorrector(horizons, latent_dim=2),
        "Kernel-H3 (RFF D=8)": HypothesisKernelH3_Corrector(horizons, rff_dim=8, seed=10),
        "Kernel-H3 (RFF D=16)": HypothesisKernelH3_Corrector(horizons, rff_dim=16, seed=42),
        "Kernel-H3 (RFF D=32)": HypothesisKernelH3_Corrector(horizons, rff_dim=32, seed=99),
    }

    crps_results = {m: {h: [] for h in horizons} for m in models}
    queue = CausalFeedbackQueue()

    for t in range(len(series) - max(horizons)):
        current_val = series[t]

        # True future values
        true_future = {h: series[t + h] for h in horizons}

        # Nonlinear shift ground truth for this timestamp
        h_shifts = {}
        for h in horizons:
            nl_shift = alpha[t] * math.sin(math.pi * math.log2(max(1, h)) / 3.0) + beta[t] * (
                (math.log2(max(1, h)) / 4.0) ** 2
            )
            h_shifts[h] = nl_shift

        # Base forecasts from frozen model (unaware of nonlinear twist)
        base_samples_dict = {}
        for h in horizons:
            disp = math.sqrt(1.0 + 0.1 * h)
            samples = np.random.normal(current_val, disp, size=100)
            base_samples_dict[h] = samples

            # Evaluate each model
            for m_name, model in models.items():
                if m_name == "Frozen Zero-Shot (B0)":
                    adapted_samples = samples
                elif m_name == "Linear Shared State (H1)":
                    # naive offset
                    adapted_samples = samples
                elif "Linear" in m_name:
                    delta_mu, scale_mult = model.get_corrections(h)
                    loc = float(np.median(samples))
                    adapted_samples = (loc + delta_mu) + scale_mult * (samples - loc)
                else:  # Kernel-H3
                    adapted_samples = model.adapt_samples(
                        samples, float(np.median(samples)), h
                    )

                c = crps_empirical_samples(
                    adapted_samples,
                    true_future[h] + h_shifts[h],
                )
                crps_results[m_name][h].append(c)

            # Enqueue causal feedback item
            queue.enqueue(
                ForecastFeedbackItem(
                    origin_t=t,
                    horizon=h,
                    target_t=t + h,
                    y_true=true_future[h] + h_shifts[h],
                    base_samples=samples,
                )
            )

        # Release feedback that has matured at t+1
        matured = queue.step_and_release(t + 1)
        if matured:
            for m_name, model in models.items():
                if model is not None:
                    model.update_with_feedback(matured, current_t=t + 1)

    # Summarize table
    summary_rows = []
    for m_name in models:
        row = {"Model": m_name}
        all_h = []
        for h in horizons:
            mean_c = float(np.mean(crps_results[m_name][h]))
            row[f"CRPS_h{h}"] = round(mean_c, 4)
            all_h.append(mean_c)
        row["Mean_CRPS"] = round(float(np.mean(all_h)), 4)
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    csv_path = ROOT_DIR / "table_kernel_benchmarks.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Exported Kernel Benchmark Table to {csv_path}")

    # Plot Figure 18
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Panel A: CRPS by Horizon
    h_labels = [f"h={h}" for h in horizons]
    x = np.arange(len(horizons))
    width = 0.16

    palette = ["#94a3b8", "#3b82f6", "#10b981", "#8b5cf6", "#ec4899"]
    plot_models = [
        "Frozen Zero-Shot (B0)",
        "Linear Unified (H3)",
        "Kernel-H3 (RFF D=8)",
        "Kernel-H3 (RFF D=16)",
        "Kernel-H3 (RFF D=32)",
    ]

    for i, m_name in enumerate(plot_models):
        vals = [df.loc[df["Model"] == m_name, f"CRPS_h{h}"].values[0] for h in horizons]
        axes[0].bar(x + (i - 2) * width, vals, width, label=m_name, color=palette[i % len(palette)])

    axes[0].set_xlabel("Forecast Horizon Lead Time (h)", fontweight="bold")
    axes[0].set_ylabel("Prequential CRPS (Lower is Better)", fontweight="bold")
    axes[0].set_title("(A) Nonlinear Term-Structure Shift: CRPS by Horizon", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(h_labels)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(frameon=True, fontsize=8)

    # Panel B: Mean CRPS vs RFF Dimension
    rff_dims = [0, 2, 8, 16, 32]
    mean_crps_curve = [
        df.loc[df["Model"] == "Frozen Zero-Shot (B0)", "Mean_CRPS"].values[0],
        df.loc[df["Model"] == "Linear Unified (H3)", "Mean_CRPS"].values[0],
        df.loc[df["Model"] == "Kernel-H3 (RFF D=8)", "Mean_CRPS"].values[0],
        df.loc[df["Model"] == "Kernel-H3 (RFF D=16)", "Mean_CRPS"].values[0],
        df.loc[df["Model"] == "Kernel-H3 (RFF D=32)", "Mean_CRPS"].values[0],
    ]
    axes[1].plot(rff_dims, mean_crps_curve, marker="o", linewidth=2.2, color="#059669")
    axes[1].fill_between(rff_dims, mean_crps_curve, max(mean_crps_curve), alpha=0.1, color="#10b981")
    axes[1].set_xlabel("RFF Representation Dimension D (0 = Frozen, 2 = Linear H3)", fontweight="bold")
    axes[1].set_ylabel("Mean CRPS across Horizons", fontweight="bold")
    axes[1].set_title("(B) Expressive Capacity vs Predictive Error", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_path = ROOT_DIR / "paper_figures" / "fig18_rff_kernel_vs_linear_manifold.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved Figure 18 to {fig_path}")


if __name__ == "__main__":
    run_benchmark()
