"""Benchmark & Visualizer for Spatio-Temporal Cross-Horizon Adaptation (ST-H3).

Generates:
1. table_spatiotemporal_benchmarks.csv
2. paper_figures/fig16_spatiotemporal_cross_horizon_transfer.png
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

from cita2027.adaptation.spatiotemporal_adaptation import (
    SpatialFeedbackItem,
    SpatioTemporalCrossHorizonAdapter,
)
from cita2027.adaptation.probabilistic_metrics import compute_crps_empirical


def run_spatiotemporal_simulation(
    num_nodes: int = 5,
    horizons: list[int] = [1, 4, 8, 16],
    length: int = 300,
    seed: int = 2027,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulates a 5-node sensor network with localized structural shock at Node 0."""
    np.random.seed(seed)

    # Adjacency: 5-node ring graph
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        A[i, (i + 1) % num_nodes] = 1.0
        A[i, (i - 1) % num_nodes] = 1.0

    st_adapter = SpatioTemporalCrossHorizonAdapter(
        num_nodes=num_nodes,
        horizons=horizons,
        adjacency_matrix=A,
        latent_dim=2,
        lambda_graph=0.25,
        lr_shared=0.08,
        lr_spatial=0.06,
    )

    # Independent adapter baseline (no graph diffusion)
    indep_adapter = SpatioTemporalCrossHorizonAdapter(
        num_nodes=num_nodes,
        horizons=horizons,
        adjacency_matrix=np.zeros_like(A),  # Independent
        latent_dim=2,
        lambda_graph=0.0,
        lr_shared=0.08,
        lr_spatial=0.0,
    )

    # Generate synthetic spatio-temporal ground truth:
    # At t = 100, node 0 suffers a permanent +3.0 mean jump and 2x volatility jump
    y = np.zeros((length, num_nodes), dtype=np.float64)
    for t in range(1, length):
        for i in range(num_nodes):
            drift = 3.0 if (i == 0 and t >= 100) else (1.5 if (i in [1, 4] and t >= 130) else 0.0)
            vol = 2.0 if (i == 0 and t >= 100) else 1.0
            y[t, i] = 0.6 * y[t - 1, i] + drift + np.random.normal(0, vol)

    # Generate base forecasts
    base_forecasts = {}
    for orig in range(length):
        base_forecasts[orig] = {}
        for i in range(num_nodes):
            base_forecasts[orig][i] = {}
            for h in horizons:
                if orig + h >= length:
                    continue
                pred_mean = (0.6 ** h) * y[orig, i]
                pred_std = np.sqrt(np.sum([0.6 ** (2 * k) for k in range(h)]))
                base_forecasts[orig][i][h] = np.random.normal(pred_mean, pred_std, size=50)

    # Causal queues for pending feedback
    pending_st: List[SpatialFeedbackItem] = []
    pending_indep: List[SpatialFeedbackItem] = []

    records = []
    state_trace = []

    for t in range(length - max(horizons)):
        # 1. Adapt and score
        for i in range(num_nodes):
            for h in horizons:
                s_base = base_forecasts[t][i][h]
                y_true = float(y[t + h, i])

                pred_st = st_adapter.adapt(i, t, h, s_base)
                pred_ind = indep_adapter.adapt(i, t, h, s_base)

                crps_st = compute_crps_empirical(pred_st, y_true)
                crps_ind = compute_crps_empirical(pred_ind, y_true)

                records.append({
                    "t": t + h,
                    "node_id": i,
                    "horizon": h,
                    "crps_st": crps_st,
                    "crps_indep": crps_ind,
                })

                # Enqueue feedback item maturing at t + h
                item = SpatialFeedbackItem(
                    node_id=i,
                    origin_t=t,
                    horizon=h,
                    target_t=t + h,
                    y_true=y_true,
                    base_samples=s_base,
                )
                pending_st.append(item)
                pending_indep.append(item)

        # 2. Release feedback items that mature at current t
        ready_st = [it for it in pending_st if it.target_t <= t]
        pending_st = [it for it in pending_st if it.target_t > t]

        ready_ind = [it for it in pending_indep if it.target_t <= t]
        pending_indep = [it for it in pending_indep if it.target_t > t]

        st_adapter.update(ready_st)
        indep_adapter.update(ready_ind)

        # Log state of nodes 0 and 1
        state_trace.append({
            "t": t,
            "node0_norm_st": float(np.linalg.norm(st_adapter.Z[0])),
            "node1_norm_st": float(np.linalg.norm(st_adapter.Z[1])),
            "node2_norm_st": float(np.linalg.norm(st_adapter.Z[2])),
            "node1_norm_indep": float(np.linalg.norm(indep_adapter.Z[1])),
        })

    return pd.DataFrame(records), pd.DataFrame(state_trace)


def plot_fig16_spatiotemporal(df_scores: pd.DataFrame, df_states: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Panel (a): Node 0 & 1 Latent State Norm Evolution
    ax_a = axes[0, 0]
    ax_a.plot(df_states["t"], df_states["node0_norm_st"], color="#ef4444", linewidth=2.0, label="Node 0 (Source of Shock)")
    ax_a.plot(df_states["t"], df_states["node1_norm_st"], color="#2563eb", linewidth=2.0, label="Node 1 with Graph Diffusion (ST-H3)")
    ax_a.plot(df_states["t"], df_states["node1_norm_indep"], color="#64748b", linestyle="--", linewidth=1.8, label="Node 1 Independent (No Diffusion)")
    ax_a.axvline(100, color="#ef4444", linestyle=":", linewidth=1.5, label="Shock at Node 0 (t=100)")
    ax_a.set_title("(a) Spatio-Temporal State Diffusion Across Network Graph", fontweight="bold")
    ax_a.set_xlabel("Prequential Time Step t")
    ax_a.set_ylabel("Latent State Norm ||z_i||")
    ax_a.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)
    ax_a.grid(True, linestyle="--", alpha=0.5)

    # Panel (b): Rolling CRPS on Neighbor Node 1 (h=4)
    ax_b = axes[0, 1]
    sub1 = df_scores[(df_scores["node_id"] == 1) & (df_scores["horizon"] == 4)].sort_values("t")
    r_st = sub1["crps_st"].rolling(20, min_periods=5).mean()
    r_ind = sub1["crps_indep"].rolling(20, min_periods=5).mean()

    ax_b.plot(sub1["t"], r_st, color="#2563eb", linewidth=2.2, label="ST-H3 (Graph Coupled)")
    ax_b.plot(sub1["t"], r_ind, color="#ef4444", linewidth=1.8, linestyle="--", label="Independent Horizon-Wise")
    ax_b.axvline(100, color="#ef4444", linestyle=":", linewidth=1.5, label="Shock at Node 0 (t=100)")
    ax_b.set_title("(b) Neighbor Node 1 Predictive Error (Horizon h=4)", fontweight="bold")
    ax_b.set_xlabel("Prequential Time Step t")
    ax_b.set_ylabel("Rolling Mean CRPS")
    ax_b.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)
    ax_b.grid(True, linestyle="--", alpha=0.5)

    # Panel (c): Horizon-wise CRPS Comparison across All Nodes
    ax_c = axes[1, 0]
    agg_h = df_scores.groupby("horizon")[["crps_st", "crps_indep"]].mean()
    horizons = agg_h.index.tolist()
    x = np.arange(len(horizons))
    width = 0.35

    ax_c.bar(x - width/2, agg_h["crps_st"], width, label="ST-H3 (Graph Laplacian)", color="#2563eb", alpha=0.9)
    ax_c.bar(x + width/2, agg_h["crps_indep"], width, label="Independent Decoupled", color="#64748b", alpha=0.9)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"h = {h}" for h in horizons])
    ax_c.set_title("(c) Mean CRPS by Forecast Horizon across 5 Nodes", fontweight="bold")
    ax_c.set_ylabel("Mean CRPS (Lower is Better)")
    ax_c.legend(frameon=True, facecolor="white", fontsize=8.5)
    ax_c.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Panel (d): Per-Node Relative Improvement
    ax_d = axes[1, 1]
    agg_node = df_scores.groupby("node_id")[["crps_st", "crps_indep"]].mean()
    gains = ((agg_node["crps_indep"] - agg_node["crps_st"]) / agg_node["crps_indep"] * 100)
    colors_nodes = ["#ef4444", "#2563eb", "#10b981", "#10b981", "#2563eb"]

    ax_d.bar(range(len(gains)), gains, color=colors_nodes, width=0.5)
    ax_d.set_xticks(range(len(gains)))
    ax_d.set_xticklabels([f"Node {i}" for i in range(len(gains))])
    ax_d.axhline(0, color="black", linewidth=1.0)
    ax_d.set_title("(d) Percentage CRPS Improvement via Spatial Coupling", fontweight="bold")
    ax_d.set_ylabel("Error Reduction (%)")
    ax_d.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.suptitle("Figure 16: Spatio-Temporal Cross-Horizon Adaptation (ST-H3) on Sensor Graph", y=1.02, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure 16 to {output_path}")


def main() -> None:
    print("=" * 80)
    print("  RUNNING SPATIO-TEMPORAL CROSS-HORIZON (ST-H3) BENCHMARK")
    print("=" * 80)

    df_scores, df_states = run_spatiotemporal_simulation(num_nodes=5, horizons=[1, 4, 8, 16], length=300, seed=2027)

    csv_path = PROJECT_ROOT / "table_spatiotemporal_benchmarks.csv"
    summary = df_scores.groupby(["node_id", "horizon"])[["crps_st", "crps_indep"]].mean().reset_index()
    summary["improvement_pct"] = (summary["crps_indep"] - summary["crps_st"]) / summary["crps_indep"] * 100
    summary.to_csv(csv_path, index=False)
    print(f"[OK] Saved table to {csv_path} ({len(summary)} rows)")

    fig_path = PROJECT_ROOT / "paper_figures" / "fig16_spatiotemporal_cross_horizon_transfer.png"
    plot_fig16_spatiotemporal(df_scores, df_states, fig_path)

    print("\n--- Spatio-Temporal Summary across Network ---")
    mean_st = df_scores["crps_st"].mean()
    mean_ind = df_scores["crps_indep"].mean()
    print(f"Overall Network Mean CRPS: ST-H3 = {mean_st:.4f} vs Independent = {mean_ind:.4f} (Gain = {(mean_ind - mean_st)/mean_ind*100:.2f}%)")


if __name__ == "__main__":
    main()
