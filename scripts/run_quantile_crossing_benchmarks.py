"""
Benchmark: Online Quantile Crossing Rectification (Quantile-H3).
Evaluates elimination of level crossing and horizon inversion violations.
Outputs:
- table_quantile_crossing.csv
- paper_figures/fig23_quantile_crossing_rectification.png (300 DPI)
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

from cita2027.adaptation.metrics import winkler_score
from cita2027.adaptation.quantile_crossing_rectifier import OnlineQuantileCrossingRectifier


def run_benchmark():
    horizons = [1, 2, 4, 8, 16]
    alphas = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    length = 300
    rng = np.random.default_rng(777)

    rectifier = OnlineQuantileCrossingRectifier(
        horizons=horizons, alpha_levels=alphas, enforce_horizon_monotonicity=True
    )

    raw_crossing_count = 0
    rectified_crossing_count = 0
    raw_inversion_count = 0
    rectified_inversion_count = 0

    raw_winkler_list = []
    rectified_winkler_list = []

    # Snapshot of quantiles for plotting
    snapshot_raw = None
    snapshot_rectified = None

    for t in range(length):
        true_val = float(rng.normal(0, 1.0))
        # Generate raw noisy empirical quantiles that exhibit crossing
        raw_dict = {}
        for h in horizons:
            loc = rng.normal(0, 0.3)
            # Add noise to standard Gaussian quantiles
            q_vals = {}
            for a in alphas:
                # Add high-frequency noise that causes crossing
                noise = rng.normal(0, 0.25)
                std_q = loc + math.sqrt(1.0 + 0.05 * h) * float(np.quantile(rng.normal(0, 1, 1000), a)) + noise
                q_vals[a] = std_q
            raw_dict[h] = q_vals

        # Check raw crossing
        had_raw_cross = False
        for h in horizons:
            vals = [raw_dict[h][a] for a in alphas]
            for i in range(len(vals) - 1):
                if vals[i] > vals[i + 1]:
                    had_raw_cross = True
                    break
        if had_raw_cross:
            raw_crossing_count += 1

        # Check raw horizon inversion (width at h=1 > width at h=16)
        w_1 = raw_dict[1][0.95] - raw_dict[1][0.05]
        w_16 = raw_dict[16][0.95] - raw_dict[16][0.05]
        if w_1 > w_16:
            raw_inversion_count += 1

        # Rectify via Quantile-H3
        rect_dict = rectifier.rectify_quantiles(raw_dict)

        # Check rectified crossing
        had_rect_cross = False
        for h in horizons:
            vals = [rect_dict[h][a] for a in alphas]
            for i in range(len(vals) - 1):
                if vals[i] > vals[i + 1] + 1e-9:
                    had_rect_cross = True
                    break
        if had_rect_cross:
            rectified_crossing_count += 1

        # Check rectified horizon inversion
        rw_1 = rect_dict[1][0.95] - rect_dict[1][0.05]
        rw_16 = rect_dict[16][0.95] - rect_dict[16][0.05]
        if rw_1 > rw_16 + 1e-9:
            rectified_inversion_count += 1

        # Winkler scores at 90% (alpha=0.10) for h=4
        w_raw = winkler_score(raw_dict[4][0.05], raw_dict[4][0.95], true_val, alpha=0.10)
        w_rect = winkler_score(rect_dict[4][0.05], rect_dict[4][0.95], true_val, alpha=0.10)
        raw_winkler_list.append(w_raw)
        rectified_winkler_list.append(w_rect)

        if t == 150:
            snapshot_raw = raw_dict
            snapshot_rectified = rect_dict

    qcvr_raw = (raw_crossing_count / float(length)) * 100.0
    qcvr_rect = (rectified_crossing_count / float(length)) * 100.0
    hir_raw = (raw_inversion_count / float(length)) * 100.0
    hir_rect = (rectified_inversion_count / float(length)) * 100.0

    table_data = [
        {
            "Method": "Raw Independent Conformal Quantiles",
            "Quantile_Crossing_Rate_Pct": round(qcvr_raw, 2),
            "Horizon_Inversion_Rate_Pct": round(hir_raw, 2),
            "Mean_Winkler_Score_h4": round(float(np.mean(raw_winkler_list)), 4),
            "Monotonicity_Guaranteed": False,
        },
        {
            "Method": "Quantile-H3 Rectified (PAVA)",
            "Quantile_Crossing_Rate_Pct": round(qcvr_rect, 2),
            "Horizon_Inversion_Rate_Pct": round(hir_rect, 2),
            "Mean_Winkler_Score_h4": round(float(np.mean(rectified_winkler_list)), 4),
            "Monotonicity_Guaranteed": True,
        },
    ]

    df = pd.DataFrame(table_data)
    csv_path = ROOT_DIR / "table_quantile_crossing.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Exported Quantile Crossing Table to {csv_path}")

    # Plot Figure 23
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Panel A: Raw Noisy Quantiles with Crossing (at t=150)
    for a in alphas:
        vals = [snapshot_raw[h][a] for h in horizons]
        axes[0].plot(horizons, vals, marker="o", label=f"alpha={a}")

    axes[0].set_xlabel("Forecast Horizon Lead Time (h)", fontweight="bold")
    axes[0].set_ylabel("Quantile Value", fontweight="bold")
    axes[0].set_title(f"(A) Raw Quantiles (Crossing Rate: {qcvr_raw:.1f}%)", fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper left", fontsize=7.5, frameon=True)

    # Panel B: Rectified Monotone Non-Crossing Quantiles
    for a in alphas:
        vals = [snapshot_rectified[h][a] for h in horizons]
        axes[1].plot(horizons, vals, marker="s", linewidth=2.0, label=f"alpha={a}")

    axes[1].set_xlabel("Forecast Horizon Lead Time (h)", fontweight="bold")
    axes[1].set_ylabel("Rectified Quantile Value", fontweight="bold")
    axes[1].set_title(f"(B) Quantile-H3 Rectified (Crossing Rate: {qcvr_rect:.1f}%)", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper left", fontsize=7.5, frameon=True)

    plt.tight_layout()
    fig_path = ROOT_DIR / "paper_figures" / "fig23_quantile_crossing_rectification.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved Figure 23 to {fig_path}")


if __name__ == "__main__":
    run_benchmark()
