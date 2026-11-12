"""Benchmark & Visualizer for Self-Tuning Online Meta-Controller (Meta-H3).

Generates:
1. table_meta_tuning.csv
2. paper_figures/fig15_online_meta_tuning_dynamics.png
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

from cita2027.adaptation.meta_tuning import MetaAdaptiveH3Controller
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.correctors import BaselineB0_Frozen, BaselineB1_IndependentEWMA
from cita2027.adaptation.synthetic_shifts import SyntheticShiftScenario
from cita2027.adaptation.feedback_queue import CausalFeedbackQueue, ForecastFeedbackItem
from cita2027.adaptation.probabilistic_metrics import compute_crps_empirical


def run_meta_simulation(
    horizons: list[int],
    length: int = 400,
    seed: int = 2027,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates Meta-H3 against static configurations under alternating non-stationary regimes."""
    scenario = SyntheticShiftScenario(horizons=horizons, seed=seed)
    # Generate multi-regime concatenated sequence: stationary -> joint_shift -> abrupt_reversal
    data_stat = scenario.generate(regime="stationary", length=150)
    data_joint = scenario.generate(regime="joint_shift", length=150)
    data_rev = scenario.generate(regime="abrupt_reversal", length=150)

    # Concatenate targets and forecasts
    y_concat = np.concatenate([data_stat["y"], data_joint["y"], data_rev["y"]])
    tot_len = len(y_concat)

    # Build concatenated base_forecasts
    base_concat = {}
    for orig in range(tot_len):
        base_concat[orig] = {}
        for h in horizons:
            if orig + h >= tot_len:
                continue
            # Sample from AR(1) stationary prior
            pred_mean = (0.7 ** h) * y_concat[orig]
            pred_std = np.sqrt(np.sum([0.7 ** (2 * k) for k in range(h)]))
            base_concat[orig][h] = np.random.normal(pred_mean, pred_std, size=50)

    meta_model = MetaAdaptiveH3Controller(horizons=horizons, eta_meta=1.0, entropy_floor=0.04)
    fixed_h3 = HypothesisH3_OverlapAwareCorrector(horizons=horizons, latent_dim=2, gamma_age=0.12, overlap_discount_factor=0.70)
    b0 = BaselineB0_Frozen(horizons=horizons)
    b1 = BaselineB1_IndependentEWMA(horizons=horizons, alpha=0.10)

    queue = CausalFeedbackQueue()
    ts_records = []
    weight_records = []

    for t in range(tot_len - max(horizons)):
        for h in horizons:
            base_s = base_concat[t][h]
            y_true = float(y_concat[t + h])

            pred_meta = meta_model.adapt_deterministic_mixture(t, h, base_s)
            pred_fixed = fixed_h3.adapt(t, h, base_s)
            pred_b0 = b0.adapt(t, h, base_s)
            pred_b1 = b1.adapt(t, h, base_s)

            crps_meta = compute_crps_empirical(pred_meta, y_true)
            crps_fixed = compute_crps_empirical(pred_fixed, y_true)
            crps_b0 = compute_crps_empirical(pred_b0, y_true)
            crps_b1 = compute_crps_empirical(pred_b1, y_true)

            if h == 4:
                ts_records.append({
                    "t": t + h,
                    "y_true": y_true,
                    "crps_meta": crps_meta,
                    "crps_fixed": crps_fixed,
                    "crps_b0": crps_b0,
                    "crps_b1": crps_b1,
                })

            queue.enqueue(
                ForecastFeedbackItem(
                    origin_t=t,
                    horizon=h,
                    target_t=t + h,
                    y_true=y_true,
                    base_samples=base_s,
                )
            )

        ready = queue.step_and_release(t)
        if ready:
            meta_model.update(ready)
            fixed_h3.update(ready)
            b1.update(ready)

            # Record weights
            w_dict = {f"w_{meta_model.configs[i].config_id}": meta_model.weights[i] for i in range(meta_model.M)}
            w_dict["t"] = t
            w_dict["entropy"] = meta_model.history_entropy[-1] if meta_model.history_entropy else 1.0
            weight_records.append(w_dict)

    df_ts = pd.DataFrame(ts_records)
    df_weights = pd.DataFrame(weight_records)
    return df_ts, df_weights


def plot_fig15_meta_tuning(df_ts: pd.DataFrame, df_weights: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
    })

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Sequence realization and regime boundaries
    ax0 = axes[0]
    ax0.plot(df_ts["t"], df_ts["y_true"], color="#334155", linewidth=1.2, label="Observation Series y_t")
    ax0.axvline(150, color="#ef4444", linestyle="--", linewidth=1.8, label="Shift: Stationary -> Joint Shift (t=150)")
    ax0.axvline(300, color="#8b5cf6", linestyle="--", linewidth=1.8, label="Shift: Joint Shift -> Abrupt Reversal (t=300)")
    ax0.set_title("(a) Multi-Regime Non-Stationary Sequence Realization", fontweight="bold")
    ax0.set_ylabel("Series Value y_t")
    ax0.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)
    ax0.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Meta-Weights evolution across portfolio
    ax1 = axes[1]
    weight_cols = [c for c in df_weights.columns if c.startswith("w_")]
    colors = ["#2563eb", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]
    for i, col in enumerate(weight_cols):
        name = col.replace("w_", "")
        ax1.plot(df_weights["t"], df_weights[col], label=name, color=colors[i % len(colors)], linewidth=2.0)
    ax1.axvline(150, color="#ef4444", linestyle="--", linewidth=1.8)
    ax1.axvline(300, color="#8b5cf6", linestyle="--", linewidth=1.8)
    ax1.set_title("(b) Online Meta-Simplex Distribution p_t(m) over Hyperparameter Configurations", fontweight="bold")
    ax1.set_ylabel("Expert Weight p_t")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Rolling CRPS performance comparison
    ax2 = axes[2]
    r_meta = df_ts["crps_meta"].rolling(20, min_periods=5).mean()
    r_fixed = df_ts["crps_fixed"].rolling(20, min_periods=5).mean()
    r_b1 = df_ts["crps_b1"].rolling(20, min_periods=5).mean()
    r_b0 = df_ts["crps_b0"].rolling(20, min_periods=5).mean()

    ax2.plot(df_ts["t"], r_meta, color="#2563eb", linewidth=2.2, label="Meta-H3 (Self-Tuning)")
    ax2.plot(df_ts["t"], r_fixed, color="#10b981", linewidth=1.6, label="Fixed H3 (gamma=0.12, beta=0.7)")
    ax2.plot(df_ts["t"], r_b1, color="#f59e0b", linewidth=1.5, label="Independent EWMA (B1)")
    ax2.plot(df_ts["t"], r_b0, color="#ef4444", linewidth=1.5, linestyle=":", label="Frozen Base (B0)")
    ax2.axvline(150, color="#ef4444", linestyle="--", linewidth=1.8)
    ax2.axvline(300, color="#8b5cf6", linestyle="--", linewidth=1.8)
    ax2.set_title("(c) Rolling Mean CRPS (Horizon h=4, Window=20)", fontweight="bold")
    ax2.set_xlabel("Prequential Time Step t")
    ax2.set_ylabel("CRPS (Lower is Better)")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Figure 15: Online Meta-Tuning (Meta-H3) Dynamics across Transitioning Regimes", y=1.02, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure 15 to {output_path}")


def main() -> None:
    print("=" * 80)
    print("  RUNNING ONLINE META-TUNING (META-H3) BENCHMARK")
    print("=" * 80)

    horizons = [1, 4, 8, 16]
    df_ts, df_weights = run_meta_simulation(horizons=horizons, length=450, seed=2027)

    csv_path = PROJECT_ROOT / "table_meta_tuning.csv"
    summary_df = pd.DataFrame([{
        "Meta_H3_Mean_CRPS": float(df_ts["crps_meta"].mean()),
        "Fixed_H3_Mean_CRPS": float(df_ts["crps_fixed"].mean()),
        "B1_EWMA_Mean_CRPS": float(df_ts["crps_b1"].mean()),
        "B0_Frozen_Mean_CRPS": float(df_ts["crps_b0"].mean()),
        "Meta_Gain_vs_B0_Percent": float((df_ts["crps_b0"].mean() - df_ts["crps_meta"].mean()) / df_ts["crps_b0"].mean() * 100),
        "Meta_Gain_vs_Fixed_Percent": float((df_ts["crps_fixed"].mean() - df_ts["crps_meta"].mean()) / df_ts["crps_fixed"].mean() * 100),
    }])
    summary_df.to_csv(csv_path, index=False)
    print(f"[OK] Saved summary table to {csv_path}")

    fig_path = PROJECT_ROOT / "paper_figures" / "fig15_online_meta_tuning_dynamics.png"
    plot_fig15_meta_tuning(df_ts, df_weights, fig_path)

    print("\n--- Meta-Tuning Performance Summary ---")
    print(summary_df.to_string())


if __name__ == "__main__":
    main()
