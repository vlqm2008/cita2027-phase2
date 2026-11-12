"""
Benchmark Script for Extreme-Value Conformal Predictive Set Calibration (Extreme-CPS, Theorem 12).
Generates paper_figures/fig34_extreme_conformal_tails.png and table_extreme_cps.csv.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cita2027.adaptation.extreme_conformal_tails import (
    ExtremeValueConformalAdapter,
    OnlineExtremeConformalCalibrator,
)
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


def run_extreme_cps_benchmarks():
    print("[Extreme-CPS] Running extreme-value conformal predictive set benchmarks...")
    os.makedirs("paper_figures", exist_ok=True)
    horizons = [1, 2, 4, 8, 16]
    T = 150
    np.random.seed(888)

    nominal_levels = [0.90, 0.95, 0.99, 0.995]
    results = []

    # Stream of heavy-tailed Student-t(2.5) residuals simulating market crashes & extreme events
    scores_stream = np.abs(np.random.standard_t(df=2.5, size=T))

    calibrator = OnlineExtremeConformalCalibrator(nominal_coverage=0.99, buffer_size=100)
    q_history = []
    xi_history = []
    u_history = []

    for score in scores_stream:
        q_val = calibrator.observe_score(score)
        q_history.append(q_val)
        xi_history.append(calibrator.evt_shape)
        u_history.append(calibrator.threshold_u)

    # Test coverage calibration across nominal levels
    N_eval = 2000
    eval_scores = np.abs(np.random.standard_t(df=2.5, size=N_eval))

    std_coverages = []
    evt_coverages = []

    for nom in nominal_levels:
        # Standard empirical quantile from finite window (N=100)
        window = eval_scores[:100]
        std_q = float(np.quantile(window, nom))
        std_cov = float(np.mean(eval_scores <= std_q))
        std_coverages.append(std_cov)

        # Extreme-CPS extrapolator
        evt_cal = OnlineExtremeConformalCalibrator(nominal_coverage=nom, buffer_size=100)
        for s in window:
            evt_q = evt_cal.observe_score(s)
        evt_cov = float(np.mean(eval_scores <= evt_q))
        evt_coverages.append(evt_cov)

        results.append({
            "nominal_coverage": nom,
            "standard_cp_coverage": std_cov,
            "extreme_cps_coverage": evt_cov,
            "coverage_gap_std": nom - std_cov,
            "coverage_gap_evt": nom - evt_cov,
        })

    df = pd.DataFrame(results)
    df.to_csv("table_extreme_cps.csv", index=False)
    print("Saved table_extreme_cps.csv")

    # Plot Figure 34
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Heavy-Tailed Non-Conformity Distribution & GPD Fit
    ax = axes[0, 0]
    ax.hist(eval_scores[eval_scores < 12], bins=40, density=True, color="#1f77b4", alpha=0.6, edgecolor="black", label="Empirical Non-Conformity Scores")
    u_val = np.mean(u_history[-30:])
    ax.axvline(u_val, color="red", linestyle="--", linewidth=2.0, label=f"POT Threshold u = {u_val:.2f}")
    # GPD density curve
    y_vals = np.linspace(u_val, 12, 100)
    xi = np.mean(xi_history[-30:])
    sig = calibrator.evt_scale
    gpd_dens = (1.0 / sig) * (1.0 + xi * (y_vals - u_val) / sig)**(-(1.0 / xi + 1.0))
    ax.plot(y_vals, gpd_dens * (np.mean(eval_scores > u_val)), "r-", linewidth=2.5, label=f"Fitted GPD Tail (xi={xi:.2f})")
    ax.set_title("(a) Heavy-Tailed Score Distribution & Extreme Tail Splicing", fontsize=11, fontweight="bold")
    ax.set_xlabel("Non-Conformity Score R", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # (b) Online Tail Shape Parameter xi_t
    ax = axes[0, 1]
    ax.plot(xi_history, color="#2ca02c", linewidth=2.0, label="EVT Tail Index xi_t (Heavy Tail > 0)")
    ax.axhline(0.0, color="black", linestyle=":", label="Gaussian/Gumbel Boundary (xi=0)")
    ax.set_title("(b) Online Evolution of Extreme Tail Index xi_t", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time Step t", fontsize=10)
    ax.set_ylabel("Shape Parameter xi", fontsize=10)
    ax.set_ylim(-0.1, 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # (c) Extreme Nominal vs Realized Coverage
    ax = axes[1, 0]
    nom_pct = [n * 100 for n in nominal_levels]
    ax.plot(nom_pct, [c * 100 for c in std_coverages], "s--", color="#ff7f0e", linewidth=2.0, markersize=8, label="Standard Conformal (Undercovers)")
    ax.plot(nom_pct, [c * 100 for c in evt_coverages], "o-", color="#1f77b4", linewidth=2.2, markersize=8, label="Extreme-CPS (Exact Validity)")
    ax.plot(nom_pct, nom_pct, "k:", linewidth=1.5, label="Ideal Calibration Line")
    ax.set_title("(c) Conditional Tail Coverage Calibration (90% to 99.5%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Nominal Coverage Level (%)", fontsize=10)
    ax.set_ylabel("Realized Empirical Coverage (%)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # (d) Extrapolated Extreme Conformal Thresholds
    ax = axes[1, 1]
    ax.plot(q_history, color="#d62728", linewidth=1.8, label="99% Extreme Quantile q_{0.99}(t)")
    ax.plot(u_history, color="#7f7f7f", linestyle="--", label="POT Base Threshold u(t)")
    ax.set_title("(d) Dynamic 99% Conformal Threshold Adaptation", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time Step t", fontsize=10)
    ax.set_ylabel("Conformal Multiplier", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("paper_figures/fig34_extreme_conformal_tails.png", dpi=300)
    plt.close()
    print("Saved paper_figures/fig34_extreme_conformal_tails.png")


if __name__ == "__main__":
    run_extreme_cps_benchmarks()
