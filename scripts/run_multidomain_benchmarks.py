"""
Execution Runner for Multi-Domain Physical Time-Series Benchmarks.

Evaluates multi-horizon online adaptation across:
1. Energy / Electricity: ETT Transformer Oil Temperature
2. Weather / Environmental: Atmospheric Surface Temperature
3. Traffic / Transportation: Highway Vehicle Flow

Follows strict prequential causal ordering without look-ahead leakage.
Generates:
- table_multidomain_benchmarks.csv
- paper_figures/fig9_multidomain_physical_benchmarks.png
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cita2027.adaptation.extended_benchmarks import ExtendedBenchmarkRunner
from cita2027.adaptation.multidomain_data import load_all_multidomain_benchmarks


def run_multidomain_benchmarks() -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("  RUNNING MULTI-DOMAIN CANONICAL PHYSICAL BENCHMARKS (ENERGY, WEATHER, TRAFFIC)")
    print("=" * 80)

    datasets = load_all_multidomain_benchmarks()
    runner = ExtendedBenchmarkRunner(horizons=[1, 4, 8, 16], num_samples=60, seed=42)
    dfs = []

    for ds in datasets:
        print(f"\nEvaluating Domain: {ds.domain} | Dataset: {ds.dataset_name} ({len(ds.values)} steps)...")
        # Run on the holdout validation + test portion
        eval_values = ds.values[ds.val_slice.start :]
        df_domain = runner.run_benchmark_on_series(eval_values, regime_name=ds.dataset_name)
        df_domain["Domain"] = ds.domain
        dfs.append(df_domain)

    df_all = pd.concat(dfs, ignore_index=True)
    out_csv = PROJECT_ROOT / "table_multidomain_benchmarks.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved multi-domain benchmarks to {out_csv.name} ({len(df_all)} rows)")

    # Print summary
    print("\n--- Multi-Domain Summary: CRPS Mean & Energy Score ---")
    summary = df_all.pivot(index="Model", columns="Regime", values=["CRPS_Mean", "Energy_Score"])
    print(summary.to_string())

    return df_all


def plot_multidomain_figure(df: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    domains = df["Regime"].unique()
    titles = [
        "(a) Energy: ETT Transformer Temperature",
        "(b) Weather: Jena Atmospheric Temperature",
        "(c) Traffic: PeMS Highway Flow",
    ]

    model_order = [
        "B0_ZeroShot", "B1_EWMA", "B2_StaticPooled", "B3_DelayAgnostic",
        "PROCEED", "PADRE", "StayCALM", "AcMCP", "RLS-TS", "H3_Unified",
    ]
    colors = [
        "#95a5a6", "#bdc3c7", "#7f8c8d", "#34495e",
        "#e67e22", "#d35400", "#9b59b6", "#2980b9", "#16a085",
        "#27ae60",
    ]

    for ax, domain_name, title in zip(axes, domains, titles):
        sub = df[df["Regime"] == domain_name].copy()
        sub = sub.set_index("Model").reindex(model_order).reset_index()

        bars = ax.barh(
            sub["Model"],
            sub["CRPS_Mean"],
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        bars[-1].set_linewidth(1.8)
        bars[-1].set_edgecolor("#1e8449")

        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + 0.01 * w,
                bar.get_y() + bar.get_height() / 2.0,
                f"{w:.3f}",
                va="center",
                ha="left",
                fontsize=8,
                fontweight="bold" if bar == bars[-1] else "normal",
            )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Mean CRPS (Lower is Better)", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.set_xlim(0, max(sub["CRPS_Mean"]) * 1.20)

    axes[0].set_ylabel("Adaptation Algorithm", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_fig = fig_dir / "fig9_multidomain_physical_benchmarks.png"
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved multi-domain figure to {out_fig.name}")


def main():
    df = run_multidomain_benchmarks()
    plot_multidomain_figure(df, PROJECT_ROOT / "paper_figures")
    print("\n[OK] Multi-domain benchmarks completed successfully.")


if __name__ == "__main__":
    main()
