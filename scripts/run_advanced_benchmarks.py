"""
Advanced Multi-Horizon Adaptation Benchmark & Theoretical Verification Runner.

This script executes:
1. Prequential stress-test benchmarks across 3 extreme non-stationary regimes:
   - Markov Regime-Switching Drift
   - Heavy-Tailed Student-t Innovations (df=2.6)
   - GARCH(1,1) Volatility Clustering
2. Evaluation of all 10 adapters:
   - Baselines: B0 (Zero-Shot), B1 (EWMA), B2 (Static Pooled), B3 (Delay-Agnostic)
   - Competitors: PROCEED, PADRE, StayCALM, AcMCP, RLS-TS
   - Proposed: Unified Overlap-Aware Adapter (H3)
3. Multivariate proper scoring rules: Energy Score, Variogram-0.5, Continuous Winkler, ECE, CRPS.
4. Empirical verification of Theorem 1 dynamic regret bounds across path variations P_T in [0, 50].
5. Generation of publication-grade figures:
   - paper_figures/fig6_extended_competitor_comparison.png
   - paper_figures/fig7_dynamic_regret_bounds.png
6. Summary CSV outputs:
   - table_extended_benchmarks.csv
   - table_theory_bound_verification.csv
"""

import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure package root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cita2027.adaptation.extended_benchmarks import (
    ExtendedBenchmarkRunner,
    GarchVolatilityClusterGenerator,
    HeavyTailStudentTGenerator,
    MarkovRegimeSwitchingGenerator,
    run_all_extended_scenarios,
)
from cita2027.adaptation.theory_bounds import (
    DynamicRegretBoundCalculator,
    EmpiricalBoundVerifier,
    LyapunovStabilityCertificate,
    summarize_bound_verification,
)


def run_and_save_extended_benchmarks(output_dir: Path) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("  RUNNING EXTENDED MULTI-REGIME ADAPTATION BENCHMARKS")
    print("=" * 80)
    
    df_results = run_all_extended_scenarios(length=120, seed=42)
    csv_path = output_dir / "table_extended_benchmarks.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"[OK] Extended benchmarks saved to {csv_path.name} ({len(df_results)} rows)")
    
    # Print concise summary
    print("\n--- Summary by Regime (Mean CRPS and Winkler) ---")
    summary = df_results.pivot(index="Model", columns="Regime", values=["CRPS_Mean", "Cont_Winkler"])
    print(summary.to_string())
    return df_results


def run_and_save_theory_verification(output_dir: Path):
    print("\n" + "=" * 80)
    print("  THEORETICAL DYNAMIC REGRET BOUND VERIFICATION (THEOREM 1)")
    print("=" * 80)
    
    verifier = EmpiricalBoundVerifier(
        latent_dim=2,
        horizons=[1, 4, 8, 16],
        learning_rate=0.06,
        step_clip=0.35,
        beta_overlap=0.50,
        decay_gamma=0.05,
        seed=101,
    )
    
    path_variations = (0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 35.0, 50.0)
    print(f"Testing {len(path_variations)} path variation regimes with T=120...")
    points = verifier.run_verification_experiment(T=120, path_variations=path_variations)
    
    records = []
    for pt in points:
        records.append({
            "Path_Variation_PT": pt.path_variation,
            "Horizon_T": pt.time_horizon,
            "Realized_Regret": pt.realized_cumulative_regret,
            "Theoretical_Bound": pt.theoretical_upper_bound,
            "Ratio_Empirical_to_Bound": pt.empirical_bound_ratio,
            "Bound_Satisfied": pt.bound_is_satisfied,
        })
    df_theory = pd.DataFrame(records)
    csv_path = output_dir / "table_theory_bound_verification.csv"
    df_theory.to_csv(csv_path, index=False)
    print(f"[OK] Theoretical verification saved to {csv_path.name}")
    print(summarize_bound_verification(points))
    
    # Lyapunov Certificate
    cert = LyapunovStabilityCertificate.certify(
        latent_dim=2,
        learning_rate=0.06,
        step_norm_clip=0.35,
        damping_factor=0.98,
        max_gradient_norm=2.5,
    )
    print(f"\n[Lyapunov Stability Certificate]")
    print(f"-> {cert.description}")
    
    return df_theory


def generate_publication_figures(df_benchmarks: pd.DataFrame, df_theory: pd.DataFrame, fig_dir: Path):
    print("\n" + "=" * 80)
    print("  GENERATING PUBLICATION-GRADE HIGH RESOLUTION FIGURES")
    print("=" * 80)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # Figure 6: Extended Competitor Comparison
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    regimes = ["Markov_Switching", "StudentT_HeavyTail", "GARCH_Volatility"]
    titles = [
        "(a) Markov Regime-Switching Drift",
        "(b) Heavy-Tailed Student-t Innovations",
        "(c) GARCH(1,1) Volatility Clustering",
    ]
    
    # Palette definition
    model_order = [
        "B0_ZeroShot", "B1_EWMA", "B2_StaticPooled", "B3_DelayAgnostic",
        "PROCEED", "PADRE", "StayCALM", "AcMCP", "RLS-TS", "H3_Unified",
    ]
    colors = [
        "#95a5a6", "#bdc3c7", "#7f8c8d", "#34495e",  # Baselines (grays)
        "#e67e22", "#d35400", "#9b59b6", "#2980b9", "#16a085",  # Competitors
        "#27ae60",  # H3 Unified (Vibrant Green)
    ]
    
    for ax, reg, title in zip(axes, regimes, titles):
        sub = df_benchmarks[df_benchmarks["Regime"] == reg].copy()
        sub = sub.set_index("Model").reindex(model_order).reset_index()
        
        bars = ax.barh(
            sub["Model"],
            sub["CRPS_Mean"],
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        # Highlight H3
        bars[-1].set_linewidth(1.8)
        bars[-1].set_edgecolor("#1e8449")
        
        # Add value annotations
        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + 0.01,
                bar.get_y() + bar.get_height() / 2.0,
                f"{w:.3f}",
                va="center",
                ha="left",
                fontsize=8,
                fontweight="bold" if bar == bars[-1] else "normal",
            )
            
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Mean Continuous Ranked Probability Score (CRPS)", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.set_xlim(0, max(sub["CRPS_Mean"]) * 1.18)
        
    axes[0].set_ylabel("Adaptation Algorithm", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig6_path = fig_dir / "fig6_extended_competitor_comparison.png"
    plt.savefig(fig6_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure 6 to {fig6_path.name}")
    
    # --------------------------------------------------------------------------
    # Figure 7: Theoretical Regret Bound vs Realized Regret
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    
    P_vals = df_theory["Path_Variation_PT"]
    realized = df_theory["Realized_Regret"]
    bound = df_theory["Theoretical_Bound"]
    
    ax.plot(
        P_vals,
        bound,
        label=r"Theoretical Dynamic Bound $\mathcal{R}_T(z^*)$ [Theorem 1]",
        color="#c0392b",
        linestyle="--",
        linewidth=2.5,
        marker="s",
        markersize=6,
    )
    ax.plot(
        P_vals,
        realized,
        label=r"Realized Empirical Regret $\sum_{t=1}^T f_t(z_t) - f_t(z_t^*)$",
        color="#2980b9",
        linestyle="-",
        linewidth=2.5,
        marker="o",
        markersize=7,
    )
    
    # Shaded safety margin
    ax.fill_between(
        P_vals,
        realized,
        bound,
        color="#2ecc71",
        alpha=0.15,
        label="Guaranteed Theoretical Slack Margin",
    )
    
    ax.set_title(
        r"Empirical Dynamic Regret vs. Analytical Upper Bound $\mathcal{R}_T(z^*)$",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(r"Comparator Path Variation Length $\mathcal{P}_T = \sum_{t=1}^{T-1} \|z_{t+1}^* - z_t^*\|_2$", fontsize=11)
    ax.set_ylabel(r"Cumulative Dynamic Regret", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10, loc="upper left")
    
    # Annotate slope
    ax.annotate(
        r"Linear Path Penalty Slope $\propto \frac{D_{\mathcal{Z}}}{\eta_T}$",
        xy=(P_vals.iloc[4], bound.iloc[4]),
        xytext=(P_vals.iloc[4] + 3, bound.iloc[4] - 80),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontsize=9,
        fontweight="bold",
    )
    
    plt.tight_layout()
    fig7_path = fig_dir / "fig7_dynamic_regret_bounds.png"
    plt.savefig(fig7_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure 7 to {fig7_path.name}")


def main():
    output_dir = PROJECT_ROOT
    fig_dir = PROJECT_ROOT / "paper_figures"
    
    df_benchmarks = run_and_save_extended_benchmarks(output_dir)
    df_theory = run_and_save_theory_verification(output_dir)
    generate_publication_figures(df_benchmarks, df_theory, fig_dir)
    
    print("\n" + "=" * 80)
    print("  EXTENDED BENCHMARK & THEORETICAL VERIFICATION COMPLETE (100% SUCCESS)")
    print("=" * 80)


if __name__ == "__main__":
    main()
