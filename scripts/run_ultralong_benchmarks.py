"""Runner script for Ultra-Long Horizon Adaptation Stress Benchmark.

Generates:
1. table_ultralong_horizons.csv
2. paper_figures/fig13_ultralong_horizon_scaling_h64.png
"""
from __future__ import annotations

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

from cita2027.adaptation.ultralong_horizons import run_ultralong_horizon_experiment


def plot_fig13_ultralong_scaling(df: pd.DataFrame, output_path: Path) -> None:
    """Generates 3-panel figure showing CRPS scaling across horizons H in {1,2,4,8,16,32,64}."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13,
    })

    regimes = ["joint_shift", "variance_shift", "abrupt_reversal"]
    regime_titles = {
        "joint_shift": "(a) Joint Mean & Volatility Shift",
        "variance_shift": "(b) Severe Volatility Explosion",
        "abrupt_reversal": "(c) Abrupt Sign Reversal",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    colors = {
        "B0_Frozen": "#6c757d",
        "B1_IndepEWMA": "#fd7e14",
        "B2_IndepLocScale": "#20c997",
        "B3_FullyPooled": "#6f42c1",
        "H1_SharedState": "#d63384",
        "H2_AgeDecay": "#0dcaf0",
        "H3_Unified": "#0d6efd",
        "H3_Dyadic": "#198754",
    }

    markers = {
        "B0_Frozen": "x",
        "B1_IndepEWMA": "s",
        "B2_IndepLocScale": "v",
        "B3_FullyPooled": "^",
        "H1_SharedState": "D",
        "H2_AgeDecay": "p",
        "H3_Unified": "o",
        "H3_Dyadic": "*",
    }

    for idx, reg in enumerate(regimes):
        ax = axes[idx]
        sub = df[df["regime"] == reg]

        for model_name in [
            "B0_Frozen",
            "B1_IndepEWMA",
            "B3_FullyPooled",
            "H1_SharedState",
            "H2_AgeDecay",
            "H3_Unified",
            "H3_Dyadic",
        ]:
            m_data = sub[sub["model"] == model_name].sort_values("horizon")
            if m_data.empty:
                continue

            linewidth = 2.4 if "H3" in model_name else 1.2
            markersize = 8 if model_name == "H3_Dyadic" else 5
            ax.plot(
                m_data["horizon"],
                m_data["crps_mean"],
                label=model_name.replace("_", " "),
                color=colors.get(model_name, "#333333"),
                marker=markers.get(model_name, "o"),
                markersize=markersize,
                linewidth=linewidth,
                alpha=0.9,
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(regime_titles.get(reg, reg), fontweight="bold")
        ax.set_xlabel("Forecast Horizon h (Log2 scale)")
        ax.set_ylabel("Mean CRPS (Lower is better)")
        ax.grid(True, linestyle="--", alpha=0.5)

        if idx == 0:
            ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")

    plt.suptitle("Figure 13: Ultra-Long Horizon Stress Scaling Benchmark (H in {1, 2, 4, 8, 16, 32, 64})", y=1.02, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure 13 to {output_path}")


def main() -> None:
    print("=" * 80)
    print("  RUNNING ULTRA-LONG HORIZON STRESS BENCHMARK (H up to 64)")
    print("=" * 80)

    horizons = [1, 2, 4, 8, 16, 32, 64]
    df = run_ultralong_horizon_experiment(horizons=horizons, length=300, seed=2027)

    csv_path = PROJECT_ROOT / "table_ultralong_horizons.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved table to {csv_path} ({len(df)} rows)")

    fig_path = PROJECT_ROOT / "paper_figures" / "fig13_ultralong_horizon_scaling_h64.png"
    plot_fig13_ultralong_scaling(df, fig_path)

    # Print summary
    print("\n--- Summary: Mean CRPS at Ultra-Long Horizons (h=32, 64) ---")
    summary = df[df["horizon"].isin([32, 64])].groupby(["horizon", "model"])["crps_mean"].mean().unstack()
    print(summary.to_string())


if __name__ == "__main__":
    main()
