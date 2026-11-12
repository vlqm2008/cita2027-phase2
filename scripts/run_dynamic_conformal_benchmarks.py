"""Evaluates Dynamic Conformal Calibrator against Static Split-Conformal.

Generates:
1. table_conformal_coverage.csv
2. paper_figures/fig14_dynamic_conformal_coverage_bands.png
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

from cita2027.adaptation.dynamic_conformal_calibrator import (
    AsynchronousCrossHorizonConformalCalibrator,
    ConformalInterval,
)
from cita2027.adaptation.synthetic_shifts import SyntheticShiftScenario
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.feedback_queue import CausalFeedbackQueue, ForecastFeedbackItem


def run_conformal_evaluation(
    horizons: list[int],
    regimes: list[str],
    alpha: float = 0.10,
    length: int = 400,
    seed: int = 2027,
) -> Tuple[pd.DataFrame, dict]:
    """Runs causal conformal calibration across regimes."""
    scenario = SyntheticShiftScenario(horizons=horizons, seed=seed)
    records = []
    trajectories = {}

    for reg in regimes:
        data = scenario.generate(regime=reg, length=length)
        model = HypothesisH3_OverlapAwareCorrector(horizons=horizons, latent_dim=2)
        calibrator_dynamic = AsynchronousCrossHorizonConformalCalibrator(
            horizons=horizons, alpha=alpha, eta_conformal=0.06, gamma_transfer=0.04
        )
        queue = CausalFeedbackQueue()

        intervals_pending: dict[Tuple[int, int], ConformalInterval] = {}
        ts_trace = []

        for t in range(length):
            # 1. Adapt and calibrate
            for h in horizons:
                if t not in data["base_forecasts"] or h not in data["base_forecasts"][t]:
                    continue
                base_samples = data["base_forecasts"][t][h]
                adapted_samples = model.adapt(t, h, base_samples)
                interval = calibrator_dynamic.calibrate_interval(t, h, adapted_samples)
                intervals_pending[(t, h)] = interval

                y_true = float(data["y"][t + h])
                queue.enqueue(
                    ForecastFeedbackItem(
                        origin_t=t,
                        horizon=h,
                        target_t=t + h,
                        y_true=y_true,
                        base_samples=base_samples,
                        adapted_samples=adapted_samples,
                    )
                )

            # 2. Release feedback and update
            ready = queue.step_and_release(t)
            if ready:
                model.update(ready)
                for item in ready:
                    inter = intervals_pending.get((item.origin_t, item.horizon))
                    if inter:
                        rec = calibrator_dynamic.update_with_feedback(item.horizon, item.y_true, inter)
                        ts_trace.append(rec)

        trajectories[reg] = pd.DataFrame(ts_trace)

        for h in horizons:
            cov = calibrator_dynamic.get_empirical_coverage(h)
            records.append({
                "regime": reg,
                "horizon": h,
                "nominal_coverage": 1.0 - alpha,
                "dynamic_conformal_coverage": cov,
                "coverage_gap": abs(cov - (1.0 - alpha)),
                "final_q": calibrator_dynamic.q[h],
            })

    return pd.DataFrame(records), trajectories


def plot_fig14_conformal_bands(trajectories: dict, output_path: Path) -> None:
    """Plots dynamic coverage tracking and conformal interval widths under abrupt shift."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    regimes_to_plot = ["joint_shift", "abrupt_reversal"]
    titles = {
        "joint_shift": "Joint Mean & Volatility Shift",
        "abrupt_reversal": "Abrupt Reversal Shift",
    }

    colors = {1: "#0d6efd", 4: "#198754", 8: "#fd7e14", 16: "#d63384"}

    for col_idx, reg in enumerate(regimes_to_plot):
        df_tr = trajectories.get(reg)
        if df_tr is None or df_tr.empty:
            continue

        ax_cov = axes[0, col_idx]
        ax_w = axes[1, col_idx]

        for h in [1, 4, 8, 16]:
            sub = df_tr[df_tr["horizon"] == h].sort_values("target_t")
            if sub.empty:
                continue

            # Rolling coverage curve
            ax_cov.plot(
                sub["target_t"],
                sub["rolling_coverage"],
                label=f"Horizon {h}",
                color=colors.get(h, "#333333"),
                linewidth=1.8,
                alpha=0.9,
            )

            # Rolling width curve
            rolling_w = sub["interval_width"].rolling(20, min_periods=5).mean()
            ax_w.plot(
                sub["target_t"],
                rolling_w,
                label=f"Horizon {h}",
                color=colors.get(h, "#333333"),
                linewidth=1.8,
                alpha=0.9,
            )

        # Target coverage line
        ax_cov.axhline(0.90, color="black", linestyle="--", linewidth=1.5, label="Target (90%)")
        ax_cov.axvline(200, color="red", linestyle=":", linewidth=1.5, label="Shift Onset (t=200)")
        ax_w.axvline(200, color="red", linestyle=":", linewidth=1.5, label="Shift Onset (t=200)")

        ax_cov.set_title(f"Rolling Empirical Coverage: {titles.get(reg, reg)}", fontweight="bold")
        ax_cov.set_ylabel("Empirical Coverage")
        ax_cov.set_ylim(0.70, 1.0)
        ax_cov.grid(True, linestyle="--", alpha=0.5)

        ax_w.set_title(f"Conformal Interval Width: {titles.get(reg, reg)}", fontweight="bold")
        ax_w.set_xlabel("Prequential Time Step t")
        ax_w.set_ylabel("Mean Interval Width")
        ax_w.grid(True, linestyle="--", alpha=0.5)

        if col_idx == 0:
            ax_cov.legend(frameon=True, facecolor="white", loc="lower left", fontsize=8.5)

    plt.suptitle("Figure 14: Asynchronous Cross-Horizon Dynamic Conformal Calibrator (ACH-DCC)", y=1.02, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure 14 to {output_path}")


def main() -> None:
    print("=" * 80)
    print("  RUNNING ASYNCHRONOUS DYNAMIC CONFORMAL BENCHMARK")
    print("=" * 80)

    horizons = [1, 4, 8, 16]
    regimes = ["joint_shift", "variance_shift", "abrupt_reversal", "mean_shift", "stationary"]

    df_results, trajectories = run_conformal_evaluation(horizons=horizons, regimes=regimes, alpha=0.10, length=400)

    csv_path = PROJECT_ROOT / "table_conformal_coverage.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"[OK] Saved table to {csv_path} ({len(df_results)} rows)")

    fig_path = PROJECT_ROOT / "paper_figures" / "fig14_dynamic_conformal_coverage_bands.png"
    plot_fig14_conformal_bands(trajectories, fig_path)

    # Print summary
    print("\n--- Summary: Empirical Coverage Across Regimes (Nominal = 0.90) ---")
    summary = df_results.pivot(index="regime", columns="horizon", values="dynamic_conformal_coverage")
    print(summary.to_string())


if __name__ == "__main__":
    main()
