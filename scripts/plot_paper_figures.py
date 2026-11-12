"""Generate publication-ready figures for the research paper."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    root = Path(__file__).resolve().parents[1]
    out_fig = root / "paper_figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    # Styling for scientific papers
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    })

    # Load Phase 3 benchmark data
    csv_path = root / "artifacts" / "adaptation" / "h3" / "h3_unified_benchmark.csv"
    if not csv_path.exists():
        print(f"Data file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # -------------------------------------------------------------
    # Figure 1: CRPS by Horizon under Joint Shift (Primary Proof)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    df_joint = df[df["regime"] == "joint_shift"]

    methods = ["B0_Frozen", "B1_IndepEWMA", "H1_Uniform", "H2_AgeDecay", "H3_Unified"]
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["x", "s", "^", "v", "o"]

    for m, c, marker in zip(methods, colors, markers):
        sub = df_joint[df_joint["method"] == m].sort_values("horizon")
        ax.plot(
            sub["horizon"],
            sub["crps"],
            label=m.replace("_", " "),
            color=c,
            marker=marker,
            linewidth=2.2,
            markersize=7,
        )

    ax.set_xlabel("Forecasting Horizon ($h$)")
    ax.set_ylabel("Continuous Ranked Probability Score (CRPS)")
    ax.set_title("Figure 1: Predictive Distribution Quality under Joint Distribution Shift")
    ax.set_xticks([1, 4, 8, 16])
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    fig.tight_layout()

    fig1_path = out_fig / "fig1_joint_shift_crps_by_horizon.png"
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Negative Transfer Reduction under Abrupt Reversal (H1 vs H2 vs H3)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    df_abrupt = df[df["regime"] == "abrupt_reversal"]

    bar_methods = ["B1_IndepEWMA", "H1_Uniform", "H2_AgeDecay", "H3_Unified"]
    sub_bar = df_abrupt[df_abrupt["method"].isin(bar_methods)]

    sns.barplot(
        data=sub_bar,
        x="horizon",
        y="crps",
        hue="method",
        palette="tab10",
        ax=ax,
    )

    ax.set_xlabel("Forecasting Horizon ($h$)")
    ax.set_ylabel("CRPS (Lower is Better)")
    ax.set_title("Figure 2: Stale Feedback Mitigation across Horizons (Abrupt Reversal)")
    ax.legend(title="Method", frameon=True)
    fig.tight_layout()

    fig2_path = out_fig / "fig2_abrupt_reversal_adaptation.png"
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: Overall Performance Across All 7 Regimes
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    piv = df.groupby(["regime", "method"])["crps"].mean().unstack()

    ordered_methods = ["B0_Frozen", "B1_IndepEWMA", "H1_Uniform", "H2_AgeDecay", "H3_Unified"]
    piv = piv[ordered_methods]

    piv.plot(kind="bar", ax=ax, colormap="viridis", width=0.82)
    ax.set_ylabel("Mean CRPS across Horizons")
    ax.set_xlabel("Distribution Shift Regime")
    ax.set_title("Figure 3: Cross-Regime Probabilistic Adaptation Evaluation")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    fig3_path = out_fig / "fig3_cross_regime_benchmark_summary.png"
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {fig3_path}")
    print("\nAll publication figures generated successfully in paper_figures/")


if __name__ == "__main__":
    main()
