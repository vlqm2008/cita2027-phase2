"""Execution runner for Real-World Multi-Horizon Adaptation Benchmark (Phase 4).

Evaluates frozen Chronos-T5-small on SPY and BTCUSDT validation splits under
asynchronous cross-horizon feedback. Runs in ultra-low resource mode (CPU, 2 threads, <400MB RAM)
so it does not impact GPU gaming or foreground apps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from cita2027.adaptation.correctors import (
    BaselineB0_Frozen,
    BaselineB1_IndependentEWMA,
    BaselineB2_IndependentLocScale,
    BaselineB3_FullyPooled,
    HypothesisH1_SharedStateCorrector,
)
from cita2027.adaptation.hypothesis_h2 import HypothesisH2_AgeWeightedCorrector
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.real_data_benchmark import (
    generate_or_load_real_forecasts,
    run_real_prequential_experiment,
    set_low_process_priority,
)


def main():
    parser = argparse.ArgumentParser(description="Real-World Multi-Horizon Adaptation Benchmark")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max-origins", type=int, default=80)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--force-regenerate", action="store_true")
    args = parser.parse_args()

    set_low_process_priority()

    root = args.root.resolve()
    out_dir = root / "artifacts" / "adaptation" / "real_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 4: REAL-WORLD MULTI-HORIZON ADAPTATION BENCHMARK (LOW-RESOURCE MODE)")
    print("=" * 80)
    print(f"Target split: {args.split}")
    print(f"Max origins: {args.max_origins}, Samples: {args.num_samples}, CPU threads: {args.num_threads}")
    print(f"Output directory: {out_dir}")

    horizons = [1, 4, 8, 16]
    datasets = ["SPY", "BTCUSDT"]

    all_results = []
    traces_to_save = {}

    for dset in datasets:
        print(f"\n[{dset}] Loading / Generating multi-horizon foundation model forecasts...")
        fc_data = generate_or_load_real_forecasts(
            dataset_name=dset,
            split=args.split,
            horizons=horizons,
            context_length=256,
            num_samples=args.num_samples,
            max_origins=args.max_origins,
            device="cpu",
            num_threads=args.num_threads,
            cache_dir=out_dir / "cache",
            root_dir=root,
            force_regenerate=args.force_regenerate,
        )

        n_origins = fc_data["N_origins"]
        print(f"[{dset}] Forecasts ready: {n_origins} evaluation origins.")

        models = {
            "B0_Frozen": BaselineB0_Frozen(horizons),
            "B1_IndepEWMA": BaselineB1_IndependentEWMA(horizons, alpha=0.10),
            "B2_IndepLocScale": BaselineB2_IndependentLocScale(horizons, alpha=0.10),
            "B3_FullyPooled": BaselineB3_FullyPooled(horizons, alpha=0.10),
            "H1_SharedState": HypothesisH1_SharedStateCorrector(horizons, latent_dim=2),
            "H2_AgeDecay": HypothesisH2_AgeWeightedCorrector(horizons, decay_type="exponential", gamma=0.12),
            "H3_Unified": HypothesisH3_OverlapAwareCorrector(
                horizons, latent_dim=2, gamma_age=0.12, overlap_discount_factor=0.7
            ),
        }

        for m_name, model in models.items():
            print(f"  Running prequential evaluation: {m_name}...")
            res = run_real_prequential_experiment(model, fc_data, alpha=0.20)
            traces_to_save[f"{dset}_{m_name}"] = res["df_trace"]

            for h in horizons:
                m = res["summary_by_horizon"][h]
                all_results.append({
                    "dataset": dset,
                    "method": m_name,
                    "horizon": h,
                    "crps": m["crps_mean"],
                    "mae": m["mae_mean"],
                    "winkler": m["winkler_mean"],
                    "width": m["width_mean"],
                    "coverage": m["coverage"],
                    "count": m["count"],
                })

    df_results = pd.DataFrame(all_results)
    results_csv = out_dir / "real_data_benchmark_results.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"\n[OK] Results saved to {results_csv}")

    # Display Pivot Summary
    for dset in datasets:
        print(f"\n==================== {dset} MULTI-HORIZON CRPS ====================")
        sub = df_results[df_results["dataset"] == dset]
        piv = sub.pivot_table(index="horizon", columns="method", values="crps")
        print(piv.round(6).to_string())

        print(f"\n==================== {dset} 80% WINKLER SCORE ====================")
        piv_w = sub.pivot_table(index="horizon", columns="method", values="winkler")
        print(piv_w.round(6).to_string())

        print(f"\n==================== {dset} 80% COVERAGE ====================")
        piv_c = sub.pivot_table(index="horizon", columns="method", values="coverage")
        print(piv_c.round(4).to_string())

    # Overall dataset average CRPS
    print("\n==================== OVERALL AVERAGE CRPS ====================")
    overall_piv = df_results.pivot_table(index="dataset", columns="method", values="crps")
    print(overall_piv.round(6).to_string())

    # Save summary json
    summary_path = out_dir / "real_data_summary.json"
    summary_dict = df_results.to_dict(orient="records")
    with open(summary_path, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"[OK] Summary JSON saved to {summary_path}")


if __name__ == "__main__":
    main()
