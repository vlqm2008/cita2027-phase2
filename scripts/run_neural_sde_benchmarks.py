"""
Benchmark Script for Continuous-Time Neural SDE Adaptation (NeuralSDE-H3, Theorem 16).
Generates paper_figures/fig39_neural_sde_continuous_time.png and table_neural_sde.csv.
"""

import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.metrics import crps_empirical_samples
from cita2027.adaptation.neural_sde_adaptation import (
    ContinuousTimeFeedbackItem,
    NeuralSDECrossHorizonAdapter,
)


def run_neural_sde_benchmarks():
    print("[NeuralSDE-H3] Running continuous-time Neural SDE benchmarks...")
    os.makedirs("paper_figures", exist_ok=True)
    horizons = [1, 2, 4, 8, 16]
    np.random.seed(999)

    sde_adapter = NeuralSDECrossHorizonAdapter(horizons=horizons, latent_dim=2, theta_drift=0.15, sigma_diffusion=0.08)

    # Generate irregular observation timestamps over continuous interval [0, 80]
    num_events = 60
    inter_arrivals = np.random.exponential(scale=1.3, size=num_events)
    event_times = np.cumsum(inter_arrivals)

    trajectory_times = []
    trajectory_z1 = []
    trajectory_z2 = []
    state_norms = []

    # True continuous drift
    true_signal_func = lambda tau: 1.5 * np.sin(tau * 0.1)

    crps_discrete = []
    crps_continuous = []

    for k, tau in enumerate(event_times):
        # Ingest feedback item
        val_true = float(true_signal_func(tau) + np.random.normal(0, 0.2))
        items = [
            ContinuousTimeFeedbackItem(
                origin_t=max(0.0, tau - 2.0),
                horizon=2,
                target_t=tau,
                y_true=val_true,
                base_samples=np.random.normal(0, 0.5, size=20),
            )
        ]

        res = sde_adapter.update_with_feedback(items, current_t=tau)

        trajectory_times.append(tau)
        trajectory_z1.append(sde_adapter.continuous_z[0])
        trajectory_z2.append(sde_adapter.continuous_z[1])
        state_norms.append(res["state_norm"])

        # Compare CRPS against discrete-step baseline
        base_test = np.random.normal(0, 0.5, size=30)
        loc_sde, _ = sde_adapter.get_corrections(2)
        crps_continuous.append(crps_empirical_samples(base_test + loc_sde, val_true))
        # Discrete baseline lagging
        loc_lag = float(true_signal_func(max(0.0, tau - 3.0)))
        crps_discrete.append(crps_empirical_samples(base_test + loc_lag, val_true))

    # Save summary table
    df = pd.DataFrame({
        "event_index": range(num_events),
        "continuous_tau": event_times,
        "state_norm": state_norms,
        "crps_continuous_sde": crps_continuous,
        "crps_discrete_baseline": crps_discrete,
    })
    df.to_csv("table_neural_sde.csv", index=False)
    print("Saved table_neural_sde.csv")

    # Plot Figure 39
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Continuous-Time Manifold Sample Path z(t)
    ax = axes[0, 0]
    ax.plot(trajectory_times, trajectory_z1, "o-", color="#1f77b4", linewidth=1.8, label="z_1(t) (Ito Drift-Diffusion)")
    ax.plot(trajectory_times, trajectory_z2, "s-", color="#ff7f0e", linewidth=1.8, label="z_2(t) (Jump Innovation)")
    for tau in event_times[::6]:
        ax.axvline(tau, color="gray", linestyle=":", alpha=0.5)
    ax.set_title("(a) Continuous-Time Latent Manifold Trajectory z(t)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Continuous Wall-Clock Timestamp tau (seconds)", fontsize=10)
    ax.set_ylabel("Latent State Magnitude", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # (b) Inter-Arrival Interval Distribution (Exponential Jitter)
    ax = axes[0, 1]
    ax.hist(inter_arrivals, bins=18, density=True, color="#2ca02c", alpha=0.7, edgecolor="black", label="Observed Inter-Arrivals Delta tau")
    x_curve = np.linspace(0, 5, 100)
    ax.plot(x_curve, (1.0 / 1.3) * np.exp(-x_curve / 1.3), "r-", linewidth=2.0, label="Theoretical Exp(lambda=0.77)")
    ax.set_title("(b) Irregular Observation Cadence Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("Elapsed Time Delta tau between Arrivals", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # (c) Continuous-Time Lyapunov Stability Envelope (Theorem 16)
    ax = axes[1, 0]
    ax.plot(trajectory_times, state_norms, color="#d62728", linewidth=2.0, label="||z(t)||_2 Realized Norm")
    ax.axhline(8.0, color="black", linestyle="--", linewidth=1.5, label="Lyapunov Ball Boundary (R=8.0)")
    lyap_bound = math.sqrt((0.08**2 * 2) / (2 * 0.15) + 4.0)
    ax.axhline(lyap_bound, color="blue", linestyle=":", label=f"Theoretical Steady State = {lyap_bound:.2f}")
    ax.set_title("(c) Continuous-Time Lyapunov Contractive Boundedness", fontsize=11, fontweight="bold")
    ax.set_xlabel("Continuous Wall-Clock Timestamp tau", fontsize=10)
    ax.set_ylabel("State Norm ||z(t)||_2", fontsize=10)
    ax.set_ylim(0, 9.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # (d) CRPS Tracking Performance under Arrival Jitter
    ax = axes[1, 1]
    ax.plot(trajectory_times, crps_discrete, color="#7f7f7f", linestyle="--", linewidth=1.5, label="Discrete Step Baseline")
    ax.plot(trajectory_times, crps_continuous, color="#1f77b4", linewidth=2.2, label="NeuralSDE-H3 (Exact Ito Integration)")
    ax.set_title("(d) Real-Time Prequential CRPS under Irregular Jitter", fontsize=11, fontweight="bold")
    ax.set_xlabel("Continuous Timestamp tau", fontsize=10)
    ax.set_ylabel("CRPS Loss", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("paper_figures/fig39_neural_sde_continuous_time.png", dpi=300)
    plt.close()
    print("Saved paper_figures/fig39_neural_sde_continuous_time.png")


if __name__ == "__main__":
    run_neural_sde_benchmarks()
