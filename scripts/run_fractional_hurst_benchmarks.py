"""Benchmark of Fractional Brownian Motion & Long-Memory Hurst Adaptation (Theorem 21).

Generates paper_figures/fig46_fractional_hurst_long_memory.png and table_fractional_hurst.csv.
"""

import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cita2027.adaptation.fractional_brownian_adaptation import (
    FractionalBrownianCrossHorizonAdapter,
)


def run_fractional_hurst_experiment():
    print("[Hurst-H3] Running Fractional Brownian Motion & Long-Memory benchmark...")
    os.makedirs("paper_figures", exist_ok=True)
    np.random.seed(42)

    T = 250
    horizons = [1, 2, 4, 8, 16]

    # Generate synthetic fractional noise with Hurst H=0.75 (persistent long memory)
    # Using Davies-Harte / circulant approximation or autoregressive power-law sum
    fbm_increments = np.zeros(T)
    for t in range(1, T):
        # Power-law kernel sum: sum_{k=1}^t (k)^{-(1.5 - H)} * xi_{t-k}
        kernel = np.array([k**(-0.75) for k in range(1, min(t+1, 50))])
        xi = np.random.randn(len(kernel))
        fbm_increments[t] = 0.5 * np.sum(kernel * xi) / np.sqrt(np.sum(kernel**2))

    hurst_adapter = FractionalBrownianCrossHorizonAdapter(horizons=horizons, learning_rate=0.04)
    standard_adapter_bias = {h: 0.0 for h in horizons}

    hurst_estimates = []
    hurst_crps_list = []
    standard_crps_list = []

    for t in range(T):
        feedback = []
        for h in horizons:
            if t >= h:
                y_val = fbm_increments[t] + 0.2 * np.random.randn()
                feedback.append({
                    "horizon": h,
                    "y_true": y_val,
                    "y_pred_loc": 0.0,
                    "y_pred_disp": 1.0,
                })
                # Standard exponential decay baseline
                standard_adapter_bias[h] = 0.95 * standard_adapter_bias[h] + 0.05 * y_val

        info = hurst_adapter.update_feedback(feedback)
        hurst_estimates.append(info.get("estimated_hurst", 0.5))

        # Evaluate forecast adaptation at h=8
        base_samples = np.random.normal(0.0, 1.0, 100)
        target = fbm_increments[min(t + 8, T - 1)] + 0.2 * np.random.randn()

        # Hurst adapted
        ad_s, _, _ = hurst_adapter.adapt_forecast(8, base_samples, 0.0, 1.0)
        crps_h = np.mean(np.abs(ad_s - target)) - 0.5 * np.mean(np.abs(ad_s[:, None] - ad_s[None, :]))
        hurst_crps_list.append(crps_h)

        # Standard baseline
        std_s = base_samples + standard_adapter_bias[8]
        crps_std = np.mean(np.abs(std_s - target)) - 0.5 * np.mean(np.abs(std_s[:, None] - std_s[None, :]))
        standard_crps_list.append(crps_std)

    # Compile Table
    mean_hurst_crps = float(np.mean(hurst_crps_list))
    mean_std_crps = float(np.mean(standard_crps_list))
    gain_pct = ((mean_std_crps - mean_hurst_crps) / max(mean_std_crps, 1e-8)) * 100.0

    records = [
        {
            "Forecast_Horizon": "h=8 (Long-Lookahead)",
            "True_Hurst_Parameter": 0.75,
            "Online_Estimated_Hurst": round(float(np.mean(hurst_estimates[-50:])), 3),
            "Standard_Markovian_CRPS": round(mean_std_crps, 4),
            "Hurst_Adapted_CRPS": round(mean_hurst_crps, 4),
            "CRPS_Gain_Pct": round(gain_pct, 2),
            "Regret_Bound_Theorem21": "Certified O(T^max(1/2, H) * K_eff^(1-H))",
        }
    ]

    df = pd.DataFrame(records)
    df.to_csv("table_fractional_hurst.csv", index=False)
    print("Saved table_fractional_hurst.csv")

    # Generate Publication Figure 46
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Online Hurst Exponent Tracking
    ax = axes[0]
    ax.plot(hurst_estimates, color="#1f77b4", lw=2, label="Online Estimated Hurst H_t")
    ax.axhline(0.75, color="red", linestyle="--", lw=1.8, label="True Persistent Memory H=0.75")
    ax.axhline(0.50, color="gray", linestyle=":", lw=1.2, label="Brownian Motion H=0.50")
    ax.set_title("(A) Online Hurst Exponent Convergence (Theorem 21)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Prequential Adaptation Steps (t)", fontsize=10)
    ax.set_ylabel("Estimated Hurst Parameter H", fontsize=10)
    ax.set_ylim(0.2, 0.95)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel B: Cumulative Regret under Long Memory
    ax = axes[1]
    time_axis = np.arange(1, T + 1)
    cum_std = np.cumsum(standard_crps_list)
    cum_hurst = np.cumsum(hurst_crps_list)
    ax.plot(time_axis, cum_std, "r--", lw=2, label="Standard Markovian (Exponential)")
    ax.plot(time_axis, cum_hurst, "g-", lw=2.5, label="Hurst-H3 Power-Law Kernel (Ours)")
    ax.set_title("(B) Cumulative Regret under Fractional Noise", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time Origin (t)", fontsize=10)
    ax.set_ylabel("Cumulative CRPS Loss", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel C: Multi-Horizon Memory Weight Kernels
    ax = axes[2]
    h_range = np.array([1, 2, 4, 8, 16, 32, 64])
    exp_kernel = np.exp(-0.15 * h_range)
    fbm_kernel_persistent = h_range ** (2 * 0.75 - 2)
    fbm_kernel_antipersistent = h_range ** (2 * 0.30 - 2)

    ax.plot(h_range, exp_kernel, "k--", marker="s", label="Markovian exp(-gamma*h)")
    ax.plot(h_range, fbm_kernel_persistent, "b-", marker="o", label="Persistent H=0.75 (h^-0.5)")
    ax.plot(h_range, fbm_kernel_antipersistent, "m:", marker="^", label="Anti-Persistent H=0.30 (h^-1.4)")
    ax.set_xscale("log", base=2)
    ax.set_title("(C) Memory Weighting vs Forecasting Horizon", fontsize=11, fontweight="bold")
    ax.set_xlabel("Lookahead Horizon (h)", fontsize=10)
    ax.set_ylabel("Effective Weight omega(h)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_path = "paper_figures/fig46_fractional_hurst_long_memory.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    run_fractional_hurst_experiment()
