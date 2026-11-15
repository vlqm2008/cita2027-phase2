"""Master End-to-End Pipeline Runner for CITA 2027 Phase 2 / Collaborator Project C+.

Executes all research phases in succession:
- Phase 1: Hypothesis H1 (Cross-Horizon Transferability)
- Phase 2: Hypothesis H2 (Stale-Feedback Negative Transfer & Age Decay)
- Phase 3: Hypothesis H3 (Overlap-Aware Stability & Unified Adapter)
- Phase 4: Real-World Time Series Benchmark (SPY & BTCUSDT)
- Phase 5: Paired Statistical Significance & Hypothesis Testing
- Phase 6: Publication-Quality Figures Generation (300 DPI)
- Phase 7: Pytest Unit Test Suite Verification
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def print_banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80 + "\n")


def run_command(cmd: list[str], desc: str, env: dict, cwd: Path | None = None) -> float:
    print_banner(f"STARTING: {desc}")
    t0 = time.time()
    res = subprocess.run(cmd, env=env, cwd=cwd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[ERROR] Step '{desc}' failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"\n[COMPLETED] {desc} in {elapsed:.2f} seconds.")
    return elapsed


def main():
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Master End-to-End Pipeline Runner")
    parser.add_argument("--root", type=Path, default=default_root)
    args = parser.parse_args()

    root = args.root.resolve()
    python_exe = sys.executable

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'};{root / 'scripts'};{env.get('PYTHONPATH', '')}"

    def run_step(cmd: list[str], desc: str) -> float:
        return run_command(cmd, desc, env, cwd=root)

    print("=" * 80)
    print("   CITA 2027: ASYNCHRONOUS CROSS-HORIZON ADAPTATION MASTER PIPELINE")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Python: {python_exe}")
    start_total = time.time()

    timings = {}

    # Phase 1: Hypothesis H1
    timings["Phase 1 (Hypothesis H1)"] = run_step(
        [python_exe, str(root / "scripts" / "run_hypothesis_h1.py"), "--root", str(root)],
        "Phase 1: Hypothesis H1 Evaluation (Cross-Horizon Transferability)",
    )

    # Phase 2: Hypothesis H2
    timings["Phase 2 (Hypothesis H2)"] = run_step(
        [python_exe, str(root / "scripts" / "run_hypothesis_h2.py"), "--root", str(root)],
        "Phase 2: Hypothesis H2 Evaluation (Stale Feedback & Age Decay)",
    )

    # Phase 3: Hypothesis H3
    timings["Phase 3 (Hypothesis H3)"] = run_step(
        [python_exe, str(root / "scripts" / "run_hypothesis_h3.py"), "--root", str(root)],
        "Phase 3: Hypothesis H3 Evaluation (Unified Cross-Horizon Benchmark)",
    )

    # Phase 4: Real-World Time Series Benchmark
    timings["Phase 4 (Real-World Benchmark)"] = run_step(
        [python_exe, str(root / "scripts" / "run_real_data_adaptation.py"), "--root", str(root)],
        "Phase 4: Real-World Multi-Horizon Adaptation (SPY & BTCUSDT)",
    )

    # Phase 5: Statistical Significance Tests
    timings["Phase 5 (Statistical Tests)"] = run_step(
        [python_exe, str(root / "scripts" / "run_statistical_tests.py"), "--root", str(root)],
        "Phase 5: Paired Statistical Significance & Hypothesis Testing",
    )

    # Phase 6: Systematic Ablation Studies
    timings["Phase 6 (Ablation Studies)"] = run_step(
        [python_exe, str(root / "scripts" / "run_ablation_study.py"), "--root", str(root)],
        "Phase 6: Systematic Ablation Studies (Components, Capacity, Sensitivity)",
    )

    # Phase 7: Publication Figures
    timings["Phase 7 (Publication Figures)"] = run_step(
        [python_exe, str(root / "scripts" / "plot_paper_figures.py"), "--root", str(root)],
        "Phase 7: Publication-Quality Figures Generation (300 DPI)",
    )

    # Phase 8: Advanced Benchmarks & Theoretical Regret Bounds
    timings["Phase 8 (Advanced Benchmarks & Theory)"] = run_step(
        [python_exe, str(root / "scripts" / "run_advanced_benchmarks.py")],
        "Phase 8: Extended Competitor Stress Tests & Theorem 1 Regret Bounds",
    )

    # Phase 9: Comprehensive Camera-Ready PDF Compilation
    timings["Phase 9 (Camera-Ready PDF)"] = run_step(
        [python_exe, str(root / "scripts" / "generate_comprehensive_paper_pdf.py")],
        "Phase 9: Comprehensive Camera-Ready PDF Compilation (Embedded 300 DPI)",
    )

    # Phase 10: Pytest Unit Tests
    timings["Phase 10 (Full Unit Test Suite)"] = run_step(
        [
            python_exe,
            "-m",
            "pytest",
            str(root / "tests" / "test_adaptation_h1.py"),
            str(root / "tests" / "test_adaptation_h2.py"),
            str(root / "tests" / "test_adaptation_h3.py"),
            str(root / "tests" / "test_real_data_benchmark.py"),
            str(root / "tests" / "test_significance.py"),
            str(root / "tests" / "test_advanced_adaptation.py"),
            str(root / "tests" / "test_handoff_architecture.py"),
            "-v",
        ],
        "Phase 10: Full Pytest Unit Test Suite Verification (40 Tests)",
    )

    total_time = time.time() - start_total

    print_banner("ALL PHASES COMPLETED WITH 100% SUCCESS")
    print(f"{'Phase':<45} | {'Duration (s)':<15}")
    print("-" * 65)
    for phase_name, duration in timings.items():
        print(f"{phase_name:<45} | {duration:>10.2f}s")
    print("-" * 65)
    print(f"{'TOTAL PIPELINE EXECUTION TIME':<45} | {total_time:>10.2f}s")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
