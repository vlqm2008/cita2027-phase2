"""
Unified Command-Line Interface (CLI) for Online Multi-Horizon Time-Series Adaptation.

Provides terminal commands:
- cita-adaptation eval-synthetic
- cita-adaptation eval-real
- cita-adaptation verify-bounds
- cita-adaptation run-pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def cmd_eval_synthetic(args: argparse.Namespace) -> None:
    from cita2027.adaptation.synthetic_shifts import SyntheticShiftScenario
    from cita2027.adaptation.extended_benchmarks import ExtendedBenchmarkRunner

    print(f"\n[CLI] Running synthetic evaluation on regime: {args.regime}")
    scenario = SyntheticShiftScenario(horizons=args.horizons, n_samples=args.samples, seed=args.seed)
    data = scenario.generate(regime=args.regime, length=args.length)
    runner = ExtendedBenchmarkRunner(horizons=args.horizons, num_samples=args.samples, seed=args.seed)
    df = runner.run_benchmark_on_series(data["y"], regime_name=args.regime)

    print("\n--- Evaluation Results ---")
    print(df[["Model", "CRPS_Mean", "Cont_Winkler", "Energy_Score", "ECE"]].to_string(index=False))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\n[OK] Results saved to {args.output}")


def cmd_eval_real(args: argparse.Namespace) -> None:
    from cita2027.adaptation.multidomain_data import load_all_multidomain_benchmarks
    from cita2027.adaptation.extended_benchmarks import ExtendedBenchmarkRunner

    print(f"\n[CLI] Running real-world evaluation on multi-domain physical datasets...")
    datasets = load_all_multidomain_benchmarks()
    runner = ExtendedBenchmarkRunner(horizons=args.horizons, num_samples=args.samples, seed=args.seed)
    dfs = []

    for ds in datasets:
        print(f"  -> Evaluating {ds.domain} / {ds.dataset_name}...")
        df_d = runner.run_benchmark_on_series(ds.values[ds.val_slice.start :], regime_name=ds.dataset_name)
        dfs.append(df_d)

    df_all = pd.concat(dfs, ignore_index=True)
    summary = df_all.pivot(index="Model", columns="Regime", values=["CRPS_Mean", "Energy_Score"])
    print("\n--- Multi-Domain Summary ---")
    print(summary.to_string())

    if args.output:
        df_all.to_csv(args.output, index=False)
        print(f"\n[OK] Results saved to {args.output}")


def cmd_verify_bounds(args: argparse.Namespace) -> None:
    from cita2027.adaptation.theory_bounds import EmpiricalBoundVerifier, summarize_bound_verification

    print(f"\n[CLI] Verifying Theorem 1 Dynamic Regret Bounds across path variations...")
    verifier = EmpiricalBoundVerifier(
        latent_dim=args.latent_dim,
        horizons=args.horizons,
        learning_rate=args.lr,
        step_clip=args.step_clip,
        seed=args.seed,
    )
    p_vals = (0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 35.0, 50.0)
    points = verifier.run_verification_experiment(T=args.length, path_variations=p_vals)
    print("\n" + summarize_bound_verification(points))


def cmd_run_pipeline(args: argparse.Namespace) -> None:
    from subprocess import run
    root = Path(__file__).resolve().parent.parent.parent.parent
    runner_script = root / "scripts" / "run_all_phases.py"
    print(f"\n[CLI] Executing master 10-phase pipeline via {runner_script.name}...")
    run([sys.executable, str(runner_script)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cita-adaptation",
        description="CLI for Asynchronous Cross-Horizon Probabilistic Adaptation for Time-Series Foundation Models",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: eval-synthetic
    p_syn = subparsers.add_parser("eval-synthetic", help="Evaluate adapters on synthetic distribution shifts")
    p_syn.add_argument("--regime", type=str, default="joint_shift", choices=[
        "stationary", "mean_shift", "variance_shift", "joint_shift",
        "shared_cross_horizon_drift", "horizon_specific_drift", "abrupt_reversal", "transient_outliers"
    ])
    p_syn.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8, 16])
    p_syn.add_argument("--samples", type=int, default=50)
    p_syn.add_argument("--length", type=int, default=120)
    p_syn.add_argument("--seed", type=int, default=42)
    p_syn.add_argument("--output", type=str, default="")
    p_syn.set_defaults(func=cmd_eval_synthetic)

    # Subcommand: eval-real
    p_real = subparsers.add_parser("eval-real", help="Evaluate adapters on real physical benchmarks")
    p_real.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8, 16])
    p_real.add_argument("--samples", type=int, default=50)
    p_real.add_argument("--seed", type=int, default=42)
    p_real.add_argument("--output", type=str, default="")
    p_real.set_defaults(func=cmd_eval_real)

    # Subcommand: verify-bounds
    p_bounds = subparsers.add_parser("verify-bounds", help="Verify Theorem 1 dynamic regret upper bounds")
    p_bounds.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8, 16])
    p_bounds.add_argument("--latent-dim", type=int, default=2)
    p_bounds.add_argument("--lr", type=float, default=0.06)
    p_bounds.add_argument("--step-clip", type=float, default=0.35)
    p_bounds.add_argument("--length", type=int, default=120)
    p_bounds.add_argument("--seed", type=int, default=101)
    p_bounds.set_defaults(func=cmd_verify_bounds)

    # Subcommand: run-pipeline
    p_pipe = subparsers.add_parser("run-pipeline", help="Execute complete 10-phase pipeline end-to-end")
    p_pipe.set_defaults(func=cmd_run_pipeline)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
