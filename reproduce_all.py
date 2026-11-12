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

    # 1. Full Pytest Suite (300+ tests across 30 test suites)
    timings["Pytest Unit Tests (301 tests)"] = run(
        [py, "-m", "pytest", "-v"],
        "Pytest Unit Tests (301 tests across 30 test suites)",
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

    # 13. Cross-Horizon Self-Attention Benchmark (Attn-H3)
    timings["Cross-Horizon Attention (Attn-H3)"] = run(
        [py, "scripts/run_attention_cross_horizon_benchmarks.py"],
        "Cross-Horizon Self-Attention Heatmaps (Attn-H3 - fig21)",
    )

    # 14. Direct Multivariate Energy Score Optimization (ES-H3)
    timings["Direct Energy Score (ES-H3)"] = run(
        [py, "scripts/run_energy_score_optimization_benchmarks.py"],
        "Direct Energy Score Online Optimization (ES-H3 - fig22)",
    )

    # 15. Online Quantile Crossing Rectification (Quantile-H3)
    timings["Quantile Crossing Rectification (Quantile-H3)"] = run(
        [py, "scripts/run_quantile_crossing_benchmarks.py"],
        "Quantile Crossing Rectification PAVA (Quantile-H3 - fig23)",
    )

    # 16. Expanded Multi-Domain Physical & Economic Evaluation
    timings["Expanded Multi-Domain (Solar, FX, Load)"] = run(
        [py, "scripts/run_multidomain_expanded_benchmarks.py"],
        "Expanded Multi-Domain Physical Benchmarks (fig24)",
    )

    # 17. Master Full-System Radar Synthesis
    timings["Master Full-System Radar Synthesis"] = run(
        [py, "scripts/run_full_system_synthesis.py"],
        "Master 6D Radar & Waterfall Synthesis (fig25)",
    )

    # 18. Multi-Agent Federated Asynchronous Consensus (Fed-ACHA)
    timings["Federated Asynchronous Adaptation (Fed-ACHA)"] = run(
        [py, "scripts/run_federated_benchmarks.py"],
        "Multi-Agent Federated Consensus (Fed-ACHA - fig26)",
    )

    # 19. Jump-Diffusion Bipower Variation Decomposition (JumpDiff-H3)
    timings["Jump-Diffusion Adaptation (JumpDiff-H3)"] = run(
        [py, "scripts/run_jump_diffusion_benchmarks.py"],
        "Jump-Diffusion Bipower Variation (JumpDiff-H3 - fig27)",
    )

    # 20. Online Granger-Causal Directed Graph Discovery (CausalGraph-H3)
    timings["Causal DAG Discovery (CausalGraph-H3)"] = run(
        [py, "scripts/run_causal_discovery_benchmarks.py"],
        "Granger-Causal Cross-Horizon Discovery (CausalGraph-H3 - fig28)",
    )

    # 21. Sequential Bayesian Online Posterior Adaptation (Bayes-H3)
    timings["Bayesian Online Posterior Adaptation (Bayes-H3)"] = run(
        [py, "scripts/run_bayesian_posterior_benchmarks.py"],
        "Sequential Bayesian Posterior Filter (Bayes-H3 - fig29)",
    )

    # 22. Definitive Grand Benchmark Atlas (Figure 30)
    timings["Definitive Grand Benchmark Atlas"] = run(
        [py, "scripts/run_atlas_benchmark.py"],
        "Grand Benchmark Atlas & 8D Radar (Figure 30 Milestone)",
    )

    # 23. Streaming Change-Point Martingale (CPM-H3)
    timings["Change-Point Martingale (CPM-H3)"] = run(
        [py, "scripts/run_cpm_benchmarks.py"],
        "Streaming Change-Point Martingale (CPM-H3 - fig31)",
    )

    # 24. Information-Geometric Riemannian Natural Gradient (Riemann-H3)
    timings["Riemannian Natural Gradient (Riemann-H3)"] = run(
        [py, "scripts/run_riemannian_benchmarks.py"],
        "Riemannian Natural Gradient Fisher Manifold (Riemann-H3 - fig32)",
    )

    # 25. Asynchronous Multi-Fidelity Cross-Horizon Adaptation (MultiFidelity-H3)
    timings["Multi-Fidelity Fusion (MultiFidelity-H3)"] = run(
        [py, "scripts/run_multifidelity_benchmarks.py"],
        "Multi-Fidelity Cross-Horizon Fusion (MultiFidelity-H3 - fig33)",
    )

    # 26. Extreme-Value Tail Conformal Predictive Sets (Extreme-CPS)
    timings["Extreme Conformal Tail Sets (Extreme-CPS)"] = run(
        [py, "scripts/run_extreme_cps_benchmarks.py"],
        "Extreme-Value Tail Conformal Predictive Sets (Extreme-CPS - fig34)",
    )

    # 27. Industrial Cyber-Physical Digital Twin Benchmark Atlas (Figure 35)
    timings["Digital Twin Benchmark Atlas (Figure 35)"] = run(
        [py, "scripts/run_digital_twin_benchmarks.py"],
        "Industrial Digital Twin Atlas & 9D Radar (Figure 35 Milestone)",
    )

    # 28. Hyperbolic Poincare Geometry Manifold Adaptation (Hyperbolic-H3)
    timings["Hyperbolic Geometry Manifold (Hyperbolic-H3)"] = run(
        [py, "scripts/run_hyperbolic_benchmarks.py"],
        "Hyperbolic Poincare Ball Geometry (Hyperbolic-H3 - fig36)",
    )

    # 29. Stochastic Delayed Feedback Game-Theoretic Minimax Equilibrium (GameTheoretic-H3)
    timings["Game-Theoretic Adversarial (GameTheoretic-H3)"] = run(
        [py, "scripts/run_game_theoretic_benchmarks.py"],
        "Game-Theoretic Minimax Adversarial Equilibrium (GameTheoretic-H3 - fig37)",
    )

    # 30. Asynchronous Non-Parametric Dirichlet Process Mixture (DPM-H3)
    timings["Dirichlet Process Mixture (DPM-H3)"] = run(
        [py, "scripts/run_dirichlet_process_benchmarks.py"],
        "Asynchronous Dirichlet Process Mixture (DPM-H3 - fig38)",
    )

    # 31. Continuous-Time Neural Stochastic Differential Equations (NeuralSDE-H3)
    timings["Neural SDE Continuous-Time (NeuralSDE-H3)"] = run(
        [py, "scripts/run_neural_sde_benchmarks.py"],
        "Continuous-Time Neural Stochastic Differential Equations (NeuralSDE-H3 - fig39)",
    )

    # 32. Definitive Grand Horizon Research Atlas (Figure 40)
    timings["Grand Horizon Research Atlas (Figure 40)"] = run(
        [py, "scripts/run_grand_horizon_atlas.py"],
        "Definitive Grand Horizon Research Atlas (Figure 40 Milestone)",
    )

    # 33. Koopman Operator Spectral Cross-Horizon Linearization (Koopman-H3)
    timings["Koopman Spectral Linearization (Koopman-H3)"] = run(
        [py, "scripts/run_koopman_benchmarks.py"],
        "Koopman Spectral Dynamic Mode Decomposition (Koopman-H3 - fig41)",
    )

    # 34. Differentially Private Cross-Horizon Adaptation (DP-ACHA)
    timings["Differential Privacy (DP-ACHA)"] = run(
        [py, "scripts/run_dp_benchmarks.py"],
        "Differentially Private Cross-Horizon Adaptation (DP-ACHA - fig42)",
    )

    # 35. Causal Invariant Risk Minimization (CausalIRM-H3)
    timings["Causal Invariant Risk Minimization (CausalIRM-H3)"] = run(
        [py, "scripts/run_causal_irm_benchmarks.py"],
        "Causal Invariant Risk Minimization for OOD Transfer (CausalIRM-H3 - fig43)",
    )

    # 36. Optimal Transport Wasserstein-2 Barycenter Morphing (Wasserstein-H3)
    timings["Wasserstein Barycenter Morphing (Wasserstein-H3)"] = run(
        [py, "scripts/run_wasserstein_benchmarks.py"],
        "Optimal Transport Wasserstein Barycenter Morphing (Wasserstein-H3 - fig44)",
    )

    # 37. Universal Grand Master Synthesis Atlas (Milestone Figure 45)
    timings["Universal Grand Master Atlas (Figure 45)"] = run(
        [py, "scripts/run_universal_master_atlas.py"],
        "Universal Grand Master Synthesis Atlas & 25-Method Taxonomy (Figure 45 Milestone)",
    )

    # 38. Scientific Integrity & Causal Provenance Audit
    timings["Scientific Integrity & Compliance Audit"] = run(
        [py, "scripts/verify_scientific_integrity.py"],
        "Comprehensive Scientific Integrity & Provenance Compliance Audit",
    )

    # 39. Master 10-Phase Pipeline
    timings["Master 10-Phase Pipeline"] = run(
        [py, "scripts/run_all_phases.py"],
        "End-to-End 10-Phase Adaptation Pipeline",
    )

    # 40. Camera-Ready PDF Compilation (45 Figures 300 DPI)
    timings["Camera-Ready PDF Compilation (45 Figures 300 DPI)"] = run(
        [py, "scripts/generate_comprehensive_paper_pdf.py"],
        "Camera-Ready PDF Compilation (45 Figures 300 DPI)",
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
