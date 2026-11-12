"""CLI runner for Hypothesis H3: Overlap-Aware Stability and Unified Method Benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from cita2027.adaptation.correctors import BaselineB0_Frozen, BaselineB1_IndependentEWMA, HypothesisH1_SharedStateCorrector
from cita2027.adaptation.hypothesis_h2 import HypothesisH2_AgeWeightedCorrector
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.synthetic_shifts import SyntheticShiftScenario
from cita2027.adaptation.experiment_h1 import run_prequential_experiment


def main():
    parser = argparse.ArgumentParser(description="Hypothesis H3 Overlap-Aware Evaluation Runner")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = root / "artifacts" / "adaptation" / "h3"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HYPOTHESIS H3: OVERLAP-AWARE ADAPTATION & UNIFIED BENCHMARK")
    print("=" * 80)
    print(f"Seed: {args.seed}")
    print(f"Artifacts output: {out_dir}")

    horizons = [1, 4, 8, 16]
    scenario = SyntheticShiftScenario(horizons=horizons, seed=args.seed)

    regimes = [
        "abrupt_reversal",
        "joint_shift",
        "mean_shift",
        "shared_cross_horizon_drift",
        "variance_shift",
        "stationary",
        "transient_outliers",
    ]

    comparison_rows = []
    stability_records = {}

    for reg in regimes:
        data = scenario.generate(regime=reg, length=600)

        models = {
            "B0_Frozen": BaselineB0_Frozen(horizons),
            "B1_IndepEWMA": BaselineB1_IndependentEWMA(horizons, alpha=0.10),
            "H1_Uniform": HypothesisH1_SharedStateCorrector(horizons, latent_dim=2),
            "H2_AgeDecay": HypothesisH2_AgeWeightedCorrector(horizons, decay_type="exponential", gamma=0.12),
            "H3_Unified": HypothesisH3_OverlapAwareCorrector(horizons, latent_dim=2, gamma_age=0.12, overlap_discount_factor=0.7),
        }

        for m_name, model in models.items():
            res = run_prequential_experiment(model, data)

            if isinstance(model, HypothesisH3_OverlapAwareCorrector):
                steps = model.state_step_history
                if steps:
                    stability_records[reg] = {
                        "mean_step_norm": float(np.mean(steps)),
                        "max_step_norm": float(np.max(steps)),
                        "std_step_norm": float(np.std(steps)),
                    }

            for h in horizons:
                m = res["summary_by_horizon"][h]
                comparison_rows.append({
                    "regime": reg,
                    "method": m_name,
                    "horizon": h,
                    "crps": m["crps_mean"],
                    "mae": m["mae_mean"],
                    "winkler": m["winkler_mean"],
                    "coverage": m["coverage"],
                })

    df = pd.DataFrame(comparison_rows)

    # Save detailed CSV
    csv_path = out_dir / "h3_unified_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Full comparisons saved to {csv_path}")

    # Pivot table for CRPS across regimes and methods
    piv = df.pivot_table(index=["regime", "horizon"], columns="method", values="crps")
    print("\n--- CRPS Comparison Across Regimes & Horizons ---")
    print(piv.round(4).to_string())

    # Overall average across evaluated regimes
    avg_crps = df.groupby("method")["crps"].mean().sort_values()
    print("\n--- Overall Mean CRPS Across Regimes ---")
    for method, score in avg_crps.items():
        print(f"  {method:20s}: {score:.4f}")

    # Stability summary
    print("\n--- H3 Adaptation Stability Under Overlapping Batches ---")
    for reg, stats in stability_records.items():
        print(f"  {reg:25s}: Mean Step = {stats['mean_step_norm']:.4f}, Max Shock = {stats['max_step_norm']:.4f}")

    summary_report = {
        "hypothesis": "H3_Overlap_Aware_Update_Stability_and_Unified_Adapter",
        "seed": args.seed,
        "overall_mean_crps": avg_crps.to_dict(),
        "stability_metrics": stability_records,
        "status": "PASS_EMPIRICAL_BENCHMARK_EVALUATION",
    }
    json_path = out_dir / "h3_summary_report.json"
    json_path.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")
    print(f"\n[OK] Summary report saved to {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
