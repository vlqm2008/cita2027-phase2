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

    # 1. Full Pytest Suite (47 tests)
    timings["Pytest Unit Tests (47 tests)"] = run(
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
            "-v",
        ],
        "Pytest Unit Tests (47 tests across 8 test suites)",
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

    # 5. Master 10-Phase Pipeline
    timings["Master 10-Phase Pipeline"] = run(
        [py, "scripts/run_all_phases.py"],
        "End-to-End 10-Phase Adaptation Pipeline",
    )

    # 6. Camera-Ready PDF Compilation
    timings["Camera-Ready PDF Compilation"] = run(
        [py, "scripts/generate_comprehensive_paper_pdf.py"],
        "Camera-Ready PDF Compilation (12 Figures 300 DPI)",
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
