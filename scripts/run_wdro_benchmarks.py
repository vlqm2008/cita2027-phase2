"""Benchmark & Visualizer for Wasserstein Distributionally Robust Adaptation (W-DRO-H3).

Generates:
1. table_wdro_benchmarks.csv
2. paper_figures/fig17_wasserstein_dro_outlier_robustness.png
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

from cita2027.adaptation.distributionally_robust_adaptation import WassersteinDRO_CrossHorizonAdapter
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.correctors import BaselineB0_Frozen, BaselineB1_IndependentEWMA
from cita2027.adaptation.synthetic_shifts import SyntheticShiftScenario
from cita2027.adaptation.feedback_queue import CausalFeedbackQueue, ForecastFeedbackItem
from cita2027.adaptation.probabilistic_metrics import compute_crps_empirical


def run_wdro_contamination_sweep(
    horizons: list[int] = [1, 4, 8, 16],
    contamination_levels: list[float] = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20],
    length: int = 300,
    seed: int = 2027,
) -> pd.DataFrame:
    """Evaluates adapters under varying contamination rates of sensor glitches."""
    scenario = SyntheticShiftScenario(horizons=horizons, seed=seed)
    data = scenario.generate(regime="joint_shift", length=length)

    records = []

    for eps in contamination_levels:
        np.random.seed(seed + int(eps * 1000))

        m_wdro = WassersteinDRO_CrossHorizonAdapter(horizons=horizons, huber_tuning_k=2.2)
        m_h3 = HypothesisH3_OverlapAwareCorrector(horizons=horizons)
        m_b1 = BaselineB1_IndependentEWMA(horizons=horizons, alpha=0.10)
        m_b0 = BaselineB0_Frozen(horizons=horizons)

        q_wdro = CausalFeedbackQueue()
        q_h3 = CausalFeedbackQueue()
        q_b1 = CausalFeedbackQueue()

        crps_wdro, crps_h3, crps_b1, crps_b0 = [], [], [], []

        for t in range(length - max(horizons)):
            for h in horizons:
                s_base = data["base_forecasts"][t][h]
                y_clean = float(data["y"][t + h])

                # Inject outlier contamination with probability eps
                is_outlier = np.random.rand() < eps
                y_obs = (y_clean + np.random.choice([-1, 1]) * np.random.uniform(15.0, 30.0)) if is_outlier else y_clean

                p_wdro = m_wdro.adapt(t, h, s_base)
                p_h3 = m_h3.adapt(t, h, s_base)
                p_b1 = m_b1.adapt(t, h, s_base)
                p_b0 = m_b0.adapt(t, h, s_base)

                # Score against CLEAN ground truth (outlier is a faulty measurement)
                crps_wdro.append(compute_crps_empirical(p_wdro, y_clean))
                crps_h3.append(compute_crps_empirical(p_h3, y_clean))
                crps_b1.append(compute_crps_empirical(p_b1, y_clean))
                crps_b0.append(compute_crps_empirical(p_b0, y_clean))

                # Queue the OBSERVED (potentially corrupted) feedback
                item = ForecastFeedbackItem(origin_t=t, horizon=h, target_t=t + h, y_true=y_obs, base_samples=s_base)
                q_wdro.enqueue(item)
                q_h3.enqueue(item)
                q_b1.enqueue(item)

            r_w = q_wdro.step_and_release(t)
            r_3 = q_h3.step_and_release(t)
            r_1 = q_b1.step_and_release(t)

            if r_w:
                m_wdro.update(r_w)
                m_h3.update(r_3)
                m_b1.update(r_1)

        records.append({
            "contamination_rate": eps,
            "CRPS_W_DRO_H3": float(np.mean(crps_wdro)),
            "CRPS_Standard_H3": float(np.mean(crps_h3)),
            "CRPS_Indep_EWMA": float(np.mean(crps_b1)),
            "CRPS_Frozen_B0": float(np.mean(crps_b0)),
            "WDRO_Advantage_vs_H3": float(np.mean(crps_h3) - np.mean(crps_wdro)),
        })

    return pd.DataFrame(records)


def plot_fig17_wdro_robustness(df: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a): CRPS Degradation Curves as Contamination Increases
    ax_a = axes[0]
    ax_a.plot(df["contamination_rate"] * 100, df["CRPS_W_DRO_H3"], marker="o", color="#2563eb", linewidth=2.4, label="Wasserstein DRO-H3 (Ours)")
    ax_a.plot(df["contamination_rate"] * 100, df["CRPS_Standard_H3"], marker="s", color="#ef4444", linewidth=1.8, linestyle="--", label="Standard Unified H3")
    ax_a.plot(df["contamination_rate"] * 100, df["CRPS_Indep_EWMA"], marker="^", color="#f59e0b", linewidth=1.8, linestyle=":", label="Independent EWMA (B1)")
    ax_a.plot(df["contamination_rate"] * 100, df["CRPS_Frozen_B0"], marker="x", color="#64748b", linewidth=1.5, linestyle="-.", label="Frozen Base (B0)")

    ax_a.set_title("(a) Mean CRPS Degradation under Feedback Contamination", fontweight="bold")
    ax_a.set_xlabel("Contamination Rate (%)")
    ax_a.set_ylabel("Clean Evaluation Mean CRPS (Lower is Better)")
    ax_a.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)
    ax_a.grid(True, linestyle="--", alpha=0.5)

    # Panel (b): Relative Advantage of W-DRO over Standard H3
    ax_b = axes[1]
    gains = (df["CRPS_Standard_H3"] - df["CRPS_W_DRO_H3"]) / df["CRPS_Standard_H3"] * 100
    colors = ["#10b981" if g > 0 else "#64748b" for g in gains]

    bars = ax_b.bar(df["contamination_rate"] * 100, gains, width=1.5, color=colors, alpha=0.9, edgecolor="#334155")
    ax_b.axhline(0, color="black", linewidth=1.0)
    ax_b.set_title("(b) Relative Error Reduction via Wasserstein DRO Clipping", fontweight="bold")
    ax_b.set_xlabel("Contamination Rate (%)")
    ax_b.set_ylabel("W-DRO-H3 Advantage over Standard H3 (%)")
    ax_b.grid(True, linestyle="--", alpha=0.5, axis="y")

    for bar in bars:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"+{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.suptitle("Figure 17: Wasserstein Distributionally Robust Adaptation (W-DRO-H3) under Outlier Contamination", y=1.02, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure 17 to {output_path}")


def main() -> None:
    print("=" * 80)
    print("  RUNNING WASSERSTEIN DRO (W-DRO-H3) CONTAMINATION BENCHMARK")
    print("=" * 80)

    df = run_wdro_contamination_sweep(
        horizons=[1, 4, 8, 16],
        contamination_levels=[0.0, 0.02, 0.05, 0.10, 0.15, 0.20],
        length=300,
        seed=2027,
    )

    csv_path = PROJECT_ROOT / "table_wdro_benchmarks.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved table to {csv_path} ({len(df)} rows)")

    fig_path = PROJECT_ROOT / "paper_figures" / "fig17_wasserstein_dro_outlier_robustness.png"
    plot_fig17_wdro_robustness(df, fig_path)

    print("\n--- Wasserstein DRO Contamination Sweep Summary ---")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
