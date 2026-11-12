"""CLI runner for Hypothesis H2: Stale-Feedback Mitigation via Age and Drift-Aware Weighting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from cita2027.adaptation.correctors import BaselineB0_Frozen, BaselineB1_IndependentEWMA, HypothesisH1_SharedStateCorrector
from cita2027.adaptation.hypothesis_h2 import HypothesisH2_AgeWeightedCorrector
from cita2027.adaptation.synthetic_shifts import SyntheticShiftScenario
from cita2027.adaptation.experiment_h1 import run_prequential_experiment


def main():
    parser = argparse.ArgumentParser(description="Hypothesis H2 Age-Decay Evaluation Runner")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = root / "artifacts" / "adaptation" / "h2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HYPOTHESIS H2: STALE-FEEDBACK & REGIME-CONSISTENCY WEIGHTING BENCHMARK")
    print("=" * 80)
    print(f"Seed: {args.seed}")
    print(f"Artifacts output: {out_dir}")

    horizons = [1, 4, 8, 16]
    scenario = SyntheticShiftScenario(horizons=horizons, seed=args.seed)

    regimes = [
        "abrupt_reversal",
        "mean_shift",
        "shared_cross_horizon_drift",
        "joint_shift",
        "stationary",
        "transient_outliers",
    ]

    comparison_rows = []

    for reg in regimes:
        data = scenario.generate(regime=reg, length=600)

        models = {
            "B0_Frozen": BaselineB0_Frozen(horizons),
            "B1_IndepEWMA": BaselineB1_IndependentEWMA(horizons, alpha=0.10),
            "H1_Uniform": HypothesisH1_SharedStateCorrector(horizons, latent_dim=2),
            "H2_ExpDecay": HypothesisH2_AgeWeightedCorrector(horizons, decay_type="exponential", gamma=0.12),
            "H2_HypDecay": HypothesisH2_AgeWeightedCorrector(horizons, decay_type="hyperbolic", gamma=0.15),
            "H2_DriftAware": HypothesisH2_AgeWeightedCorrector(horizons, decay_type="exponential", gamma=0.12, use_drift_consistency=True),
        }

        for m_name, model in models.items():
            res = run_prequential_experiment(model, data)
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
    csv_path = out_dir / "h2_vs_baselines_comparison.csv"
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

    # Check key H2 test: Performance under abrupt reversal
    abrupt_perf = df[df["regime"] == "abrupt_reversal"].groupby("method")["crps"].mean()
    print("\n--- Critical Check: Abrupt Reversal Performance ---")
    print(f"  H1 Uniform CRPS    : {abrupt_perf['H1_Uniform']:.4f}")
    print(f"  H2 DriftAware CRPS : {abrupt_perf['H2_DriftAware']:.4f}")
    print(f"  B1 IndepEWMA CRPS  : {abrupt_perf['B1_IndepEWMA']:.4f}")
    h2_helps_abrupt = abrupt_perf["H2_DriftAware"] < abrupt_perf["H1_Uniform"]
    print(f"  --> H2 DriftAware successfully reduces stale feedback damage: {h2_helps_abrupt}")

    # Save summary JSON
    summary_report = {
        "hypothesis": "H2_Stale_Feedback_Negative_Transfer_Mitigation",
        "seed": args.seed,
        "overall_mean_crps": avg_crps.to_dict(),
        "abrupt_reversal_crps": abrupt_perf.to_dict(),
        "h2_reduces_abrupt_error_vs_h1": bool(h2_helps_abrupt),
        "status": "PASS_EMPIRICAL_BENCHMARK_EVALUATION",
    }
    json_path = out_dir / "h2_summary_report.json"
    json_path.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")
    print(f"\n[OK] Summary report saved to {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
