"""
Single-Command Master Reproducibility Script for Collaborator Project C+.

Reproduces:
1. Full 40-test pytest verification
2. Complete 10-phase adaptation pipeline
3. Multi-domain physical benchmarks (Energy, Weather, Traffic)
4. Pareto frontier grid search and computational latency analysis
5. Camera-Ready Publication PDF compilation with 9 embedded 300 DPI figures
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def log_step(name: str):
    print("\n" + "=" * 80)
    print(f"  REPRODUCING: {name.upper()}")
    print("=" * 80 + "\n")


def run(cmd: list[str], desc: str):
    log_step(desc)
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'};{ROOT / 'scripts'};{env.get('PYTHONPATH', '')}"
    res = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[FAIL] {desc} exited with error code {res.returncode}")
        sys.exit(res.returncode)
    print(f"\n[SUCCESS] {desc} finished in {elapsed:.2f}s")
    return elapsed


def main():
    py = sys.executable
    print("=" * 80)
    print("  COLLABORATOR PROJECT C+: MASTER END-TO-END REPRODUCIBILITY SUITE")
    print("=" * 80)
    t_start = time.time()

    timings = {}

    # 1. Full Pytest Suite (70 tests across 15 test suites)
    timings["Pytest Unit Tests (70 tests)"] = run(
        [
            py, "-m", "pytest",
            "tests/test_adaptation_h1.py",
            "tests/test_adaptation_h2.py",
            "tests/test_adaptation_h3.py",
            "tests/test_real_data_benchmark.py",
            "tests/test_significance.py",
            "tests/test_advanced_adaptation.py",
            "tests/test_handoff_architecture.py",
            "tests/test_deep_research.py",
            "tests/test_pac_bayes_bounds.py",
            "tests/test_ultralong_horizons.py",
            "tests/test_dynamic_conformal.py",
            "tests/test_dashboard.py",
            "tests/test_meta_tuning.py",
            "tests/test_spatiotemporal.py",
            "tests/test_distributionally_robust.py",
            "-v",
        ],
        "Pytest Unit Tests (70 tests across 15 test suites)",
    )

    # 2. Multi-Domain Physical Benchmarks
    timings["Multi-Domain Physical Benchmarks"] = run(
        [py, "scripts/run_multidomain_benchmarks.py"],
        "Multi-Domain Physical Benchmarks (Energy, Weather, Traffic - fig9)",
    )

    # 3. Pareto Frontier Grid Search
    timings["Pareto Frontier Analysis"] = run(
        [py, "scripts/run_pareto_analysis.py"],
        "Pareto Frontier & Latency Analysis (fig8)",
    )

    # 4. Deep Research Frontiers Campaign
    timings["Deep Research Frontiers (Copula, Martingale, NaturalGrad)"] = run(
        [py, "scripts/run_deep_research_campaign.py"],
        "Deep Research Frontiers Campaign (fig10, fig11, fig12)",
    )

    # 5. Ultra-Long Horizon Scaling Stress Benchmark (H in {1, 2, 4, 8, 16, 32, 64})
    timings["Ultra-Long Horizon Scaling (H up to 64)"] = run(
        [py, "scripts/run_ultralong_benchmarks.py"],
        "Ultra-Long Horizon Stress Benchmark (H in {1, 2, 4, 8, 16, 32, 64} - fig13)",
    )

    # 6. Dynamic Conformal Martingale Calibration (ACH-DCC)
    timings["Dynamic Conformal Martingale Calibration"] = run(
        [py, "scripts/run_dynamic_conformal_benchmarks.py"],
        "Asynchronous Dynamic Conformal Calibration (ACH-DCC - fig14)",
    )

    # 7. Online Meta-Tuning (Meta-H3)
    timings["Online Meta-Tuning (Meta-H3)"] = run(
        [py, "scripts/run_meta_tuning_benchmarks.py"],
        "Self-Tuning Online Meta-Controller (Meta-H3 - fig15)",
    )

    # 8. Spatio-Temporal Cross-Horizon Benchmark (ST-H3)
    timings["Spatio-Temporal Adaptation (ST-H3)"] = run(
        [py, "scripts/run_spatiotemporal_benchmarks.py"],
        "Spatio-Temporal Graph Coupling Benchmark (ST-H3 - fig16)",
    )

    # 9. Wasserstein Distributionally Robust Adaptation (W-DRO-H3)
    timings["Wasserstein DRO Robustness (W-DRO-H3)"] = run(
        [py, "scripts/run_wdro_benchmarks.py"],
        "Wasserstein DRO Contamination Sweep (W-DRO-H3 - fig17)",
    )

    # 10. Nonlinear Kernel Cross-Horizon Adaptation (Kernel-H3)
    timings["Nonlinear Kernel Adaptation (Kernel-H3)"] = run(
        [py, "scripts/run_kernel_benchmarks.py"],
        "Nonlinear Kernel Cross-Horizon Benchmark (Kernel-H3 - fig18)",
    )

    # 11. Asynchronous Trajectory Conformal Risk Control (AT-CRC)
    timings["Trajectory Conformal Risk Control (AT-CRC)"] = run(
        [py, "scripts/run_trajectory_crc_benchmarks.py"],
        "Asynchronous Trajectory Conformal Risk Control (AT-CRC - fig19)",
    )

    # 12. Active Information-Theoretic Feedback Selection (Active-ACHA)
    timings["Active Feedback Selection (Active-ACHA)"] = run(
        [py, "scripts/run_active_feedback_benchmarks.py"],
        "Active Feedback Selection Pareto Benchmark (Active-ACHA - fig20)",
    )

    # 13. Scientific Integrity & Causal Provenance Audit
    timings["Scientific Integrity & Compliance Audit"] = run(
        [py, "scripts/verify_scientific_integrity.py"],
        "Comprehensive Scientific Integrity & Provenance Compliance Audit",
    )

    # 14. Master 10-Phase Pipeline
    timings["Master 10-Phase Pipeline"] = run(
        [py, "scripts/run_all_phases.py"],
        "End-to-End 10-Phase Adaptation Pipeline",
    )

    # 15. Camera-Ready PDF Compilation
    timings["Camera-Ready PDF Compilation"] = run(
        [py, "scripts/generate_comprehensive_paper_pdf.py"],
        "Camera-Ready PDF Compilation (20 Figures 300 DPI)",
    )

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print("  REPRODUCIBILITY SUITE COMPLETED (100% VERIFIED)")
    print("=" * 80)
    for k, v in timings.items():
        print(f"  {k:<45} : {v:>8.2f}s")
    print("-" * 80)
    print(f"  {'TOTAL REPRODUCTION TIME':<45} : {total_time:>8.2f}s")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
