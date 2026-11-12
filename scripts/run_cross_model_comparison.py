"""Cross-Architecture Foundation Model Generalization Benchmark.

Compares frozen zero-shot and adapted performance across two distinct TSFM architectures:
1. Amazon Chronos-T5-small (Autoregressive tokenized language model)
2. Salesforce Moirai-1.0-R-small (Masked patch transformer)
Across SPY (S&P 500 ETF) and BTCUSDT (Binance spot crypto).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def evaluate_stream(
    y: np.ndarray,
    q10_raw: np.ndarray,
    q50_raw: np.ndarray,
    q90_raw: np.ndarray,
    alpha: float = 0.20,
) -> dict[str, dict[str, float]]:
    """Simulate online prequential adaptation on rolling quantile streams."""
    n = len(y)
    models = ["B0_Frozen", "B1_IndepEWMA", "B2_LocScale", "H3_Unified"]

    # State variables
    b1_bias = 0.0
    b2_loc = 0.0
    b2_log_scale = 0.0
    h3_bias = 0.0
    h3_log_scale = 0.0

    lr_b1 = 0.10
    lr_b2_loc = 0.10
    lr_b2_scale = 0.05
    lr_h3 = 0.08

    results = {m: {"mae": [], "winkler": [], "coverage": [], "width": []} for m in models}

    for t in range(n):
        y_t = y[t]

        # 1. Predictions for step t
        preds = {
            "B0_Frozen": (q10_raw[t], q50_raw[t], q90_raw[t]),
            "B1_IndepEWMA": (q10_raw[t] + b1_bias, q50_raw[t] + b1_bias, q90_raw[t] + b1_bias),
            "B2_LocScale": (
                q50_raw[t] + b2_loc + np.exp(b2_log_scale) * (q10_raw[t] - q50_raw[t]),
                q50_raw[t] + b2_loc,
                q50_raw[t] + b2_loc + np.exp(b2_log_scale) * (q90_raw[t] - q50_raw[t]),
            ),
            "H3_Unified": (
                q50_raw[t] + h3_bias + np.exp(h3_log_scale) * (q10_raw[t] - q50_raw[t]),
                q50_raw[t] + h3_bias,
                q50_raw[t] + h3_bias + np.exp(h3_log_scale) * (q90_raw[t] - q50_raw[t]),
            ),
        }

        # 2. Evaluate metrics
        for m, (lo, med, hi) in preds.items():
            mae = abs(y_t - med)
            width = hi - lo
            penalty = (2.0 / alpha) * (max(0.0, lo - y_t) + max(0.0, y_t - hi))
            winkler = width + penalty
            cov = 1.0 if (lo <= y_t <= hi) else 0.0

            results[m]["mae"].append(mae)
            results[m]["winkler"].append(winkler)
            results[m]["coverage"].append(cov)
            results[m]["width"].append(width)

        # 3. Causal update for next step
        err = y_t - q50_raw[t]

        # B1 update
        b1_bias += lr_b1 * (y_t - (q50_raw[t] + b1_bias))

        # B2 update
        pred_med_b2 = q50_raw[t] + b2_loc
        err_b2 = y_t - pred_med_b2
        b2_loc += lr_b2_loc * err_b2
        empirical_dev = abs(err_b2) / (np.exp(b2_log_scale) * (q90_raw[t] - q10_raw[t]) / 2.56 + 1e-6)
        b2_log_scale = np.clip(b2_log_scale + lr_b2_scale * (np.log(max(1e-4, empirical_dev))), -1.5, 1.5)

        # H3 update (regime-consistent bounded update)
        pred_med_h3 = q50_raw[t] + h3_bias
        err_h3 = y_t - pred_med_h3
        drift_weight = max(0.2, min(1.0, 1.0 - abs(err_h3) / (abs(y_t) + 1e-4)))
        step_loc = np.clip(lr_h3 * drift_weight * err_h3, -0.01, 0.01)
        h3_bias += step_loc
        h3_log_scale = np.clip(h3_log_scale + 0.04 * drift_weight * (np.log(max(1e-4, abs(err_h3) / (abs(q90_raw[t] - q10_raw[t]) / 2.56 + 1e-6)))), -1.2, 1.2)

    summary = {}
    for m in models:
        summary[m] = {
            "mae": float(np.mean(results[m]["mae"])),
            "winkler": float(np.mean(results[m]["winkler"])),
            "coverage": float(np.mean(results[m]["coverage"])),
            "width": float(np.mean(results[m]["width"])),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Cross-Architecture Foundation Model Benchmark")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.root.resolve()
    tables_dir = root / "paper_tables"
    fig_dir = root / "paper_figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CROSS-ARCHITECTURE BENCHMARK: CHRONOS-T5 vs MOIRAI-1.0-R")
    print("=" * 80)

    datasets = [
        ("SPY",
         root / "artifacts" / "forecast_cache" / "validation" / "runs" / "f574fd5be7e6537c263aa8c9a646dbb8676d87a25b38a70c83131cb0e894fe54" / "chronos" / "predictions.csv",
         root / "artifacts" / "forecast_cache" / "validation" / "runs" / "f574fd5be7e6537c263aa8c9a646dbb8676d87a25b38a70c83131cb0e894fe54" / "moirai" / "predictions.csv"),
        ("BTCUSDT",
         root / "artifacts" / "forecast_cache" / "btcusdt" / "validation" / "runs" / "2a023e69a46aa952ca0015c2d8a87574616366a016c5cad333204418bda12e84" / "chronos" / "predictions.csv",
         root / "artifacts" / "forecast_cache" / "btcusdt" / "validation" / "runs" / "2a023e69a46aa952ca0015c2d8a87574616366a016c5cad333204418bda12e84" / "moirai" / "predictions.csv"),
    ]

    records = []

    for dset_name, chronos_path, moirai_path in datasets:
        print(f"\nProcessing {dset_name}...")
        df_chr = pd.read_csv(chronos_path)
        df_moi = pd.read_csv(moirai_path)

        for arch_name, df_model in [("Chronos-T5-small", df_chr), ("Moirai-1.0-R-small", df_moi)]:
            y = df_model["target_log_return"].to_numpy()
            q10 = df_model["q10"].to_numpy()
            q50 = df_model["q50"].to_numpy()
            q90 = df_model["q90"].to_numpy()

            summary = evaluate_stream(y, q10, q50, q90, alpha=0.20)
            for m_name, m_dict in summary.items():
                records.append({
                    "dataset": dset_name,
                    "backbone": arch_name,
                    "method": m_name,
                    "mae": m_dict["mae"],
                    "winkler": m_dict["winkler"],
                    "coverage": m_dict["coverage"],
                    "width": m_dict["width"],
                })

    df_res = pd.DataFrame(records)
    csv_out = tables_dir / "table_cross_architecture_comparison.csv"
    df_res.to_csv(csv_out, index=False)
    print(f"\n[OK] Results saved to {csv_out}")

    print("\n--- Summary Table ---")
    piv = df_res.pivot_table(index=["dataset", "backbone"], columns="method", values=["winkler", "coverage"])
    print(piv.round(4).to_string())

    # Generate Figure 5: Publication 300 DPI Cross-Architecture Figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Plot SPY Winkler
    spy_sub = df_res[df_res["dataset"] == "SPY"]
    piv_spy = spy_sub.pivot(index="method", columns="backbone", values="winkler")
    piv_spy.plot(kind="bar", ax=axes[0], colormap="viridis", width=0.7, edgecolor="black")
    axes[0].set_title("(a) SPY 80% Winkler Score (Lower = Better)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Winkler Score", fontsize=10)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(title="Foundation Model", frameon=True)

    # Plot BTCUSDT Winkler
    btc_sub = df_res[df_res["dataset"] == "BTCUSDT"]
    piv_btc = btc_sub.pivot(index="method", columns="backbone", values="winkler")
    piv_btc.plot(kind="bar", ax=axes[1], colormap="magma", width=0.7, edgecolor="black")
    axes[1].set_title("(b) BTCUSDT 80% Winkler Score (Lower = Better)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Winkler Score", fontsize=10)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(title="Foundation Model", frameon=True)

    plt.suptitle("Cross-Architecture Adaptation Generalization: Chronos-T5 vs. Moirai-1.0-R", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_out = fig_dir / "fig5_cross_architecture_generalization.png"
    fig.savefig(fig_out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Saved Figure 5 to {fig_out}")


if __name__ == "__main__":
    main()
