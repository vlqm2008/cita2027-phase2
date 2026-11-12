"""
Benchmark: Active Information-Theoretic Feedback Selection (Active-ACHA).
Sweeps communication/verification budgets and evaluates Pareto efficiency.
Outputs:
- table_active_feedback.csv
- paper_figures/fig20_active_feedback_budget_pareto.png (300 DPI)
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

from cita2027.adaptation.active_feedback_selection import ActiveFeedbackSelector
from cita2027.adaptation.feedback_queue import CausalFeedbackQueue, ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.metrics import crps_empirical_samples


def run_benchmark():
    horizons = [1, 2, 4, 8, 16]
    length = 300
    rng = np.random.default_rng(123)

    # Underlying series with non-stationary drift
    series = np.cumsum(rng.normal(0, 0.5, size=length))
    drift = np.zeros(length)
    drift[100:220] = 3.0  # Drift shift

    budgets = [0.8, 1.5, 2.5, 3.5, 6.0]
    strategies = [
        ("Active Information Gain (Active-ACHA)", "active_information_gain", "#10b981"),
        ("Greedy Short-Horizon", "greedy_short_horizon", "#3b82f6"),
        ("Random Uniform Subsampling", "random_uniform", "#f59e0b"),
    ]

    results_table = []
    pareto_data = {s_label: {"budgets": [], "crps": [], "cost": []} for s_label, _, _ in strategies}

    # Reference: Frozen baseline
    frozen_crps_list = []
    for t in range(length - max(horizons)):
        for h in horizons:
            disp = math.sqrt(1.0 + 0.1 * h)
            samples = rng.normal(series[t], disp, size=100)
            c = crps_empirical_samples(samples, series[t + h] + drift[t + h])
            frozen_crps_list.append(c)
    frozen_crps = float(np.mean(frozen_crps_list))

    # Reference: Full Unconstrained Feedback H3
    full_adapter = HypothesisH3_OverlapAwareCorrector(horizons, latent_dim=2)
    full_queue = CausalFeedbackQueue()
    full_crps_list = []
    for t in range(length - max(horizons)):
        for h in horizons:
            d_mu, s_mult = full_adapter.get_corrections(h)
            loc = series[t]
            disp = math.sqrt(1.0 + 0.1 * h)
            samples = rng.normal(loc, disp, size=100)
            adapted_samples = (loc + d_mu) + s_mult * (samples - loc)
            c = crps_empirical_samples(adapted_samples, series[t + h] + drift[t + h])
            full_crps_list.append(c)
            full_queue.enqueue(
                ForecastFeedbackItem(origin_t=t, horizon=h, target_t=t + h, y_true=series[t + h] + drift[t + h], base_samples=samples)
            )
        matured = full_queue.step_and_release(t + 1)
        if matured:
            full_adapter.update_with_feedback(matured, current_t=t + 1)
    full_crps = float(np.mean(full_crps_list))

    selector = ActiveFeedbackSelector(cost_policy="linear_horizon", age_decay_gamma=0.08)

    for budget in budgets:
        for s_label, s_code, _ in strategies:
            adapter = HypothesisH3_OverlapAwareCorrector(horizons, latent_dim=2)
            queue = CausalFeedbackQueue()
            crps_records = []
            cost_records = []

            for t in range(length - max(horizons)):
                for h in horizons:
                    d_mu, s_mult = adapter.get_corrections(h)
                    loc = series[t]
                    disp = math.sqrt(1.0 + 0.1 * h)
                    samples = rng.normal(loc, disp, size=100)
                    adapted_samples = (loc + d_mu) + s_mult * (samples - loc)
                    c = crps_empirical_samples(adapted_samples, series[t + h] + drift[t + h])
                    crps_records.append(c)
                    queue.enqueue(
                        ForecastFeedbackItem(origin_t=t, horizon=h, target_t=t + h, y_true=series[t + h] + drift[t + h], base_samples=samples)
                    )

                matured = queue.step_and_release(t + 1)
                if matured:
                    selected_items, cost_info = selector.select_feedback(
                        matured, current_t=t + 1, budget=budget, strategy=s_code
                    )
                    cost_records.append(cost_info["total_cost"])
                    if selected_items:
                        adapter.update_with_feedback(selected_items, current_t=t + 1)

            mean_c = float(np.mean(crps_records))
            mean_cost = float(np.mean(cost_records)) if cost_records else 0.0

            pareto_data[s_label]["budgets"].append(budget)
            pareto_data[s_label]["crps"].append(mean_c)
            pareto_data[s_label]["cost"].append(mean_cost)

            results_table.append({
                "Strategy": s_label,
                "Budget_Constraint": budget,
                "Realized_Mean_Cost": round(mean_cost, 4),
                "Mean_CRPS": round(mean_c, 4),
                "Relative_Improvement_vs_Frozen": round((frozen_crps - mean_c) / frozen_crps * 100.0, 2),
            })

    df = pd.DataFrame(results_table)
    csv_path = ROOT_DIR / "table_active_feedback.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Exported Active Feedback Table to {csv_path}")

    # Plot Figure 20
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Panel A: CRPS vs Budget Constraint
    for s_label, _, color in strategies:
        axes[0].plot(
            pareto_data[s_label]["budgets"],
            pareto_data[s_label]["crps"],
            marker="o",
            linewidth=2.2,
            label=s_label,
            color=color,
        )

    axes[0].axhline(frozen_crps, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Frozen Baseline ({frozen_crps:.3f})")
    axes[0].axhline(full_crps, color="#0f172a", linestyle=":", linewidth=1.5, label=f"Full Unconstrained H3 ({full_crps:.3f})")
    axes[0].set_xlabel("Verification Cost Budget B (per step)", fontweight="bold")
    axes[0].set_ylabel("Mean Prequential CRPS", fontweight="bold")
    axes[0].set_title("(A) Predictive CRPS vs Feedback Verification Budget", fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(frameon=True, fontsize=8.5)

    # Panel B: Efficiency Gain (CRPS vs Realized Cost)
    for s_label, _, color in strategies:
        axes[1].plot(
            pareto_data[s_label]["cost"],
            pareto_data[s_label]["crps"],
            marker="s",
            linewidth=2.2,
            label=s_label,
            color=color,
        )

    axes[1].axhline(full_crps, color="#0f172a", linestyle=":", linewidth=1.5, label="Full Budget")
    axes[1].set_xlabel("Realized Communication Cost (Units/Step)", fontweight="bold")
    axes[1].set_ylabel("Mean Prequential CRPS", fontweight="bold")
    axes[1].set_title("(B) Active Information Pareto Frontier", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(frameon=True, fontsize=8.5)

    plt.tight_layout()
    fig_path = ROOT_DIR / "paper_figures" / "fig20_active_feedback_budget_pareto.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved Figure 20 to {fig_path}")


if __name__ == "__main__":
    run_benchmark()
