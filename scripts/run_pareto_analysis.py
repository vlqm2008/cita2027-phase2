"""
Pareto Frontier and Computational Trade-off Analysis.

Explores the multi-dimensional parameter space:
- Latent state dimension d in {1, 2, 4, 8}
- Age decay parameter gamma in {0.03, 0.08, 0.15}
- Overlap factor beta in {0.35, 0.70}

Directly benchmarks the H3 adapter with the exact hyperparameter configurations:
Measures computational latency (microseconds per prequential step)
versus predictive quality (Mean CRPS across horizons {1, 4, 8, 16}).
Generates:
- table_pareto_frontier.csv
- paper_figures/fig8_pareto_frontier_tradeoff.png
"""

from __future__ import annotations

import itertools
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.multi_horizon_forecast import MultiHorizonForecaster
from cita2027.adaptation.prequential_replay import PrequentialReplayEngine
from cita2027.adaptation.probabilistic_metrics import compute_univariate_crps
from cita2027.adaptation.synthetic_shifts import SyntheticShiftScenario


def run_pareto_grid_search() -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("  RUNNING PARETO FRONTIER GRID SEARCH & LATENCY BENCHMARK")
    print("=" * 80)

    # Use Joint Shift benchmark series
    scenario = SyntheticShiftScenario(horizons=[1, 4, 8, 16], n_samples=50, seed=42)
    data = scenario.generate(regime="joint_shift", length=140)
    joint_series = data["y"]

    dims = [1, 2, 4, 8]
    gammas = [0.03, 0.08, 0.15]
    betas = [0.35, 0.70]

    configs = list(itertools.product(dims, gammas, betas))
    print(f"Directly evaluating {len(configs)} H3 configurations on Joint Shift...")

    horizons = [1, 4, 8, 16]
    forecaster = MultiHorizonForecaster(horizons=horizons, num_samples=50, model_id="Chronos-Mock")
    records = []

    for d, gamma, beta in configs:
        adapter = HypothesisH3_OverlapAwareCorrector(
            horizons=horizons,
            latent_dim=d,
            lr_shared=0.08,
            lr_local=0.03,
            gamma_age=gamma,
            overlap_discount_factor=beta,
        )

        engine = PrequentialReplayEngine(forecaster, adapter, horizons)

        t0 = time.perf_counter()
        results = engine.run_prequential_sequence(joint_series, context_window=20, dataset_name="ParetoJoint")
        elapsed = time.perf_counter() - t0

        # Compute CRPS across all steps and horizons
        all_crps = []
        for step_res in results:
            for h in horizons:
                y = step_res.y_true_horizon[h]
                samples = step_res.adapted_forecasts[h]
                crps_val = compute_univariate_crps(y, samples)
                all_crps.append(crps_val)

        mean_crps = float(np.mean(all_crps))
        num_steps = len(results)
        latency_us = (elapsed / num_steps) * 1e6

        records.append({
            "latent_dim_d": d,
            "gamma_age": gamma,
            "beta_overlap": beta,
            "latency_us_per_step": latency_us,
            "CRPS_Mean": mean_crps,
        })

    df_pareto = pd.DataFrame(records)
    out_csv = PROJECT_ROOT / "table_pareto_frontier.csv"
    df_pareto.to_csv(out_csv, index=False)
    print(f"[OK] Pareto grid search saved to {out_csv.name} ({len(df_pareto)} rows)")

    top5 = df_pareto.sort_values(by="CRPS_Mean").head(5)
    print("\n--- Top 5 Configurations by Predictive Quality ---")
    print(top5[["latent_dim_d", "gamma_age", "beta_overlap", "CRPS_Mean", "latency_us_per_step"]].to_string())

    return df_pareto


def plot_pareto_frontier(df: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    dim_colors = {1: "#3498db", 2: "#2ecc71", 4: "#e67e22", 8: "#e74c3c"}

    for d in sorted(df["latent_dim_d"].unique()):
        sub = df[df["latent_dim_d"] == d]
        ax.scatter(
            sub["latency_us_per_step"],
            sub["CRPS_Mean"],
            s=90,
            color=dim_colors[d],
            label=f"Latent Dim d={d}",
            alpha=0.85,
            edgecolors="black",
            linewidth=0.8,
        )

    # Sort for Pareto boundary
    sorted_df = df.sort_values(by=["latency_us_per_step", "CRPS_Mean"])
    pareto_pts = []
    min_crps_so_far = float("inf")

    for _, row in sorted_df.iterrows():
        if row["CRPS_Mean"] < min_crps_so_far:
            pareto_pts.append(row)
            min_crps_so_far = row["CRPS_Mean"]

    pareto_df = pd.DataFrame(pareto_pts)
    ax.plot(
        pareto_df["latency_us_per_step"],
        pareto_df["CRPS_Mean"],
        color="#8e44ad",
        linestyle="--",
        linewidth=2,
        label="Pareto Optimal Frontier",
    )

    best_pt = df.loc[df["CRPS_Mean"].idxmin()]
    ax.scatter(
        [best_pt["latency_us_per_step"]],
        [best_pt["CRPS_Mean"]],
        s=220,
        color="#f1c40f",
        edgecolors="black",
        linewidth=2,
        marker="*",
        zorder=10,
        label=f"Sweet Spot (d={int(best_pt['latent_dim_d'])}, $\\gamma$={best_pt['gamma_age']}, $\\beta$={best_pt['beta_overlap']})",
    )

    ax.set_title("Pareto Efficiency: Computational Latency vs. Probabilistic Quality", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(r"Computational Latency per Prequential Step ($\mu s$)", fontsize=11)
    ax.set_ylabel("Mean Continuous Ranked Probability Score (CRPS)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9.5, loc="upper right")

    plt.tight_layout()
    out_fig = fig_dir / "fig8_pareto_frontier_tradeoff.png"
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Pareto frontier figure to {out_fig.name}")


def main():
    df = run_pareto_grid_search()
    plot_pareto_frontier(df, PROJECT_ROOT / "paper_figures")
    print("\n[OK] Pareto frontier analysis complete.")


if __name__ == "__main__":
    main()
