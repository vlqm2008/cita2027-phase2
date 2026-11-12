"""Comprehensive Ablation Study for Asynchronous Cross-Horizon Adaptation.

Evaluates:
1. Component ablation: Bias-only vs Scale-only vs Joint adaptation
2. Latent capacity: d in {1, 2, 4, 8}
3. Age decay sensitivity: gamma in {0.0, 0.05, 0.12, 0.25, 0.50}
4. Overlap discount sensitivity: beta in {0.0, 0.3, 0.5, 0.7, 1.0}
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cita2027.adaptation.correctors import (
    BaselineB0_Frozen,
    BaselineB1_IndependentEWMA,
    BaselineB2_IndependentLocScale,
    HypothesisH1_SharedStateCorrector,
)
from cita2027.adaptation.hypothesis_h2 import HypothesisH2_AgeWeightedCorrector
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.synthetic_shifts import SyntheticShiftScenario
from cita2027.adaptation.experiment_h1 import run_prequential_experiment


def run_component_ablation(scenario_data: dict, horizons: list[int]) -> pd.DataFrame:
    """Ablation 1: Bias-only vs Scale-only vs Joint Location-Scale."""
    print("  Running Component Breakdown Ablation...")

    class ScaleOnlyCorrector(HypothesisH3_OverlapAwareCorrector):
        def adapt(self, origin_t: int, horizon: int, base_samples: np.ndarray) -> np.ndarray:
            s = np.asarray(base_samples, dtype=np.float64)
            loc = float(np.median(s))
            log_scale = float(np.dot(self.w[horizon], self.shared_scale_state) + self.local_log_scale[horizon])
            scale_mult = float(np.exp(np.clip(log_scale, -1.6, 1.6)))
            return loc + scale_mult * (s - loc)

    class BiasOnlyCorrector(HypothesisH3_OverlapAwareCorrector):
        def adapt(self, origin_t: int, horizon: int, base_samples: np.ndarray) -> np.ndarray:
            s = np.asarray(base_samples, dtype=np.float64)
            loc = float(np.median(s))
            bias = float(np.dot(self.w[horizon], self.shared_state) + self.local_bias[horizon])
            return loc + bias + (s - loc)

    models = {
        "H3_Joint": HypothesisH3_OverlapAwareCorrector(horizons, latent_dim=2, gamma_age=0.12, overlap_discount_factor=0.7),
        "H3_BiasOnly": BiasOnlyCorrector(horizons, latent_dim=2, gamma_age=0.12, overlap_discount_factor=0.7),
        "H3_ScaleOnly": ScaleOnlyCorrector(horizons, latent_dim=2, gamma_age=0.12, overlap_discount_factor=0.7),
    }

    rows = []
    for m_name, model in models.items():
        res = run_prequential_experiment(model, scenario_data)
        for h in horizons:
            rows.append({
                "variant": m_name,
                "horizon": h,
                "crps": res["summary_by_horizon"][h]["crps_mean"],
                "winkler": res["summary_by_horizon"][h]["winkler_mean"],
                "coverage": res["summary_by_horizon"][h]["coverage"],
            })
    return pd.DataFrame(rows)


def run_latent_capacity_ablation(scenario_data: dict, horizons: list[int]) -> pd.DataFrame:
    """Ablation 2: Latent State Dimension d in {1, 2, 4, 8}."""
    print("  Running Latent State Capacity Ablation...")
    dims = [1, 2, 4, 8]
    rows = []
    for d in dims:
        model = HypothesisH3_OverlapAwareCorrector(horizons, latent_dim=d, gamma_age=0.12, overlap_discount_factor=0.7)
        res = run_prequential_experiment(model, scenario_data)
        for h in horizons:
            rows.append({
                "latent_dim": d,
                "horizon": h,
                "crps": res["summary_by_horizon"][h]["crps_mean"],
                "winkler": res["summary_by_horizon"][h]["winkler_mean"],
            })
    return pd.DataFrame(rows)


def run_hyperparameter_sensitivity(scenario_data: dict, horizons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ablation 3 & 4: Gamma (age decay) and Beta (overlap discount)."""
    print("  Running Hyperparameter Sensitivity Ablations (Gamma & Beta)...")

    # Gamma sensitivity
    gammas = [0.0, 0.05, 0.12, 0.25, 0.50]
    gamma_rows = []
    for g in gammas:
        model = HypothesisH3_OverlapAwareCorrector(horizons, latent_dim=2, gamma_age=g, overlap_discount_factor=0.7)
        res = run_prequential_experiment(model, scenario_data)
        for h in horizons:
            gamma_rows.append({
                "gamma": g,
                "horizon": h,
                "crps": res["summary_by_horizon"][h]["crps_mean"],
            })

    # Beta sensitivity (overlap discount)
    betas = [0.0, 0.3, 0.5, 0.7, 1.0]
    beta_rows = []
    for b in betas:
        model = HypothesisH3_OverlapAwareCorrector(horizons, latent_dim=2, gamma_age=0.12, overlap_discount_factor=b)
        res = run_prequential_experiment(model, scenario_data)
        for h in horizons:
            beta_rows.append({
                "overlap_factor": b,
                "horizon": h,
                "crps": res["summary_by_horizon"][h]["crps_mean"],
            })

    return pd.DataFrame(gamma_rows), pd.DataFrame(beta_rows)


def plot_ablation_figure(
    df_comp: pd.DataFrame,
    df_lat: pd.DataFrame,
    df_gam: pd.DataFrame,
    df_bet: pd.DataFrame,
    out_path: Path,
) -> None:
    """Generate Figure 4: Publication 300 DPI 4-panel ablation figure."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

    # Panel A: Component breakdown
    piv_comp = df_comp.pivot_table(index="horizon", columns="variant", values="crps")
    piv_comp.plot(kind="bar", ax=axes[0, 0], colormap="viridis", width=0.75, edgecolor="black", linewidth=0.7)
    axes[0, 0].set_title("(a) Component Breakdown (Joint vs Bias vs Scale)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Mean CRPS (Lower = Better)", fontsize=10)
    axes[0, 0].set_xlabel("Forecast Horizon (h)", fontsize=10)
    axes[0, 0].legend(title="Variant", frameon=True)
    axes[0, 0].tick_params(axis="x", rotation=0)

    # Panel B: Latent Capacity
    for h in [1, 4, 8, 16]:
        sub = df_lat[df_lat["horizon"] == h]
        axes[0, 1].plot(sub["latent_dim"], sub["crps"], marker="o", label=f"h={h}", linewidth=1.8)
    axes[0, 1].set_title("(b) Latent State Capacity (Dimension d)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Latent Dimension d", fontsize=10)
    axes[0, 1].set_ylabel("Mean CRPS", fontsize=10)
    axes[0, 1].legend(title="Horizon", frameon=True)

    # Panel C: Gamma Sensitivity (Age Decay)
    for h in [1, 4, 8, 16]:
        sub = df_gam[df_gam["horizon"] == h]
        axes[1, 0].plot(sub["gamma"], sub["crps"], marker="s", label=f"h={h}", linewidth=1.8)
    axes[1, 0].set_title("(c) Stale-Feedback Decay Rate (Gamma)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Decay Parameter Gamma", fontsize=10)
    axes[1, 0].set_ylabel("Mean CRPS", fontsize=10)
    axes[1, 0].legend(title="Horizon", frameon=True)

    # Panel D: Overlap Discount Factor
    for h in [1, 4, 8, 16]:
        sub = df_bet[df_bet["horizon"] == h]
        axes[1, 1].plot(sub["overlap_factor"], sub["crps"], marker="^", label=f"h={h}", linewidth=1.8)
    axes[1, 1].set_title("(d) Overlap Effective-Sample Factor", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Overlap Discount Factor Beta", fontsize=10)
    axes[1, 1].set_ylabel("Mean CRPS", fontsize=10)
    axes[1, 1].legend(title="Horizon", frameon=True)

    plt.suptitle("Sensitivity Analysis & Ablation Studies of Unified Adapter (H3)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Saved Figure 4 to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Full Ablation Studies")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.root.resolve()
    tables_dir = root / "paper_tables"
    fig_dir = root / "paper_figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RUNNING COMPREHENSIVE ABLATION STUDIES (H1 - H3)")
    print("=" * 80)

    horizons = [1, 4, 8, 16]
    scenario = SyntheticShiftScenario(horizons=horizons, seed=2027)
    data = scenario.generate(regime="joint_shift", length=600)

    # 1. Component breakdown
    df_comp = run_component_ablation(data, horizons)
    df_comp.to_csv(tables_dir / "table_ablation_components.csv", index=False)

    # 2. Latent capacity
    df_lat = run_latent_capacity_ablation(data, horizons)
    df_lat.to_csv(tables_dir / "table_ablation_latent_dim.csv", index=False)

    # 3 & 4. Hyperparameters
    df_gam, df_bet = run_hyperparameter_sensitivity(data, horizons)
    df_gam.to_csv(tables_dir / "table_ablation_gamma.csv", index=False)
    df_bet.to_csv(tables_dir / "table_ablation_overlap.csv", index=False)

    # Plot Figure 4
    plot_ablation_figure(df_comp, df_lat, df_gam, df_bet, fig_dir / "fig4_ablation_sensitivity_curves.png")

    print("\n[OK] Ablation study complete! All tables and Figure 4 generated.")


if __name__ == "__main__":
    main()
