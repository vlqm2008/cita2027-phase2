"""
Comprehensive Test Suite for Advanced Multi-Horizon Adaptation Components.

Covers:
1. Competitor Adapters (PROCEED, PADRE, StayCALM, AcMCP, RLS-TS)
2. Distribution Metrics (Energy Score, Variogram, Continuous Winkler, PIT, ECE)
3. Theoretical Regret Bounds & Lyapunov Stability Certificates
4. Extended Synthetic Shift Generators & Prequential Integration
"""

import math
import numpy as np
import pytest

from cita2027.adaptation.competitors import (
    AcMcpAdapter,
    ForecastFeedback,
    PadreProbabilistic,
    ProceedProbabilistic,
    RecursiveLeastSquaresAdapter,
    StayCalmAggregator,
    get_all_competitors,
)
from cita2027.adaptation.distribution_metrics import (
    compute_pit_values,
    continuous_winkler_score,
    energy_score,
    evaluate_calibration_and_sharpness,
    pit_anderson_darling_uniformity,
    variogram_score,
    winkler_score_level,
)
from cita2027.adaptation.extended_benchmarks import (
    ExtendedBenchmarkRunner,
    GarchVolatilityClusterGenerator,
    HeavyTailStudentTGenerator,
    MarkovRegimeSwitchingGenerator,
)
from cita2027.adaptation.theory_bounds import (
    DynamicRegretBoundCalculator,
    EmpiricalBoundVerifier,
    LyapunovStabilityCertificate,
)


# ==============================================================================
# 1. Tests for Competitor Adapters
# ==============================================================================

def test_proceed_adapter_lifecycle():
    horizons = [1, 4, 8, 16]
    model = ProceedProbabilistic(horizons, buffer_capacity=32, n_prototypes=3)
    base_samples = np.random.normal(0.0, 1.0, size=50)

    # Prediction before any update should equal base centered
    pred0 = model.predict(4, base_samples, current_step=0)
    assert pred0.shape == (50,)
    assert np.all(np.isfinite(pred0))

    # Simulate causal arrival
    fb = [
        ForecastFeedback(
            origin_step=0,
            arrival_step=4,
            horizon=4,
            y_true=2.5,
            y_pred_samples=pred0,
            base_samples=base_samples,
        )
    ]
    model.update(fb, current_step=4)
    assert len(model.prototypes) > 0
    assert len(model.memory_buffer) == 1

    # Predict again: location should shift towards 2.5
    pred1 = model.predict(4, base_samples, current_step=5)
    assert np.mean(pred1) > np.mean(pred0)

    # Test reset
    model.reset()
    assert len(model.prototypes) == 0
    assert len(model.memory_buffer) == 0


def test_padre_adapter_correlation_tracking():
    horizons = [1, 4]
    model = PadreProbabilistic(horizons, max_lag=8)
    base_s = np.zeros(30)

    fb_batch = [
        ForecastFeedback(0, 1, 1, 1.0, base_s, base_s),
        ForecastFeedback(0, 4, 4, 1.2, base_s, base_s),
    ]
    model.update(fb_batch, current_step=4)

    assert (1, 4) in model.horizon_corr
    pred = model.predict(4, base_s, current_step=5)
    assert np.mean(pred) > 0.0


def test_staycalm_aggregator_weights_sum_to_one():
    horizons = [1, 4, 8]
    model = StayCalmAggregator(horizons)
    base_s = np.random.normal(0.0, 1.0, size=40)

    for h in horizons:
        w = model.weights[h]
        assert len(w) == model.n_experts
        assert math.isclose(float(np.sum(w)), 1.0, rel_tol=1e-5)

    # Perform updates with large positive error
    fb = [ForecastFeedback(0, 4, 4, 5.0, base_s, base_s)]
    model.update(fb, current_step=4)

    # Weights must still be normalized probability distribution
    w_after = model.weights[4]
    assert math.isclose(float(np.sum(w_after)), 1.0, rel_tol=1e-5)
    assert np.all(w_after >= 0.0)


def test_acmcp_adapter_quantile_expansion():
    horizons = [1, 4]
    model = AcMcpAdapter(horizons, target_coverage=0.90, gamma_step=0.2)
    base_s = np.random.normal(0.0, 1.0, size=50)

    # Feed an extreme outlier that breaches coverage
    fb = [ForecastFeedback(0, 1, 1, 10.0, base_s, base_s)]
    model.update(fb, current_step=1)

    # Upper quantile margin should expand
    assert model.q_upper[1] > 0.0

    # Prediction should have higher dispersion
    pred = model.predict(1, base_s, current_step=2)
    assert np.std(pred) >= np.std(base_s)


def test_rls_adapter_sherman_morrison():
    horizons = [1, 4, 8]
    model = RecursiveLeastSquaresAdapter(horizons, forgetting_factor=0.95)
    base_s = np.zeros(25)

    # Covariance matrix P must start positive definite
    assert np.all(np.linalg.eigvals(model.P) > 0)

    fb = [ForecastFeedback(0, 4, 4, 2.0, base_s, base_s)]
    model.update(fb, current_step=4)

    # Parameter vector theta should be updated
    assert np.linalg.norm(model.theta) > 0.0
    # Covariance matrix P remains symmetric
    assert np.allclose(model.P, model.P.T, atol=1e-6)


def test_get_all_competitors_factory():
    horizons = [1, 4, 8, 16]
    comps = get_all_competitors(horizons)
    assert len(comps) == 5
    for name in ["PROCEED", "PADRE", "StayCALM", "AcMCP", "RLS-TS"]:
        assert name in comps


# ==============================================================================
# 2. Tests for Distribution Metrics & Diagnostics
# ==============================================================================

def test_energy_score_properties():
    # If prediction distribution is centered exactly around truth with zero variance,
    # ES should equal 0.
    y_true = np.array([2.0, -1.0, 0.5])
    # Exact point distribution repeated S times
    trajs_exact = np.tile(y_true, (50, 1))
    es_zero = energy_score(y_true, trajs_exact)
    assert math.isclose(es_zero, 0.0, abs_tol=1e-8)

    # If prediction is shifted far from truth, ES must be strictly positive
    trajs_bad = trajs_exact + 10.0
    es_bad = energy_score(y_true, trajs_bad)
    assert es_bad > 5.0


def test_variogram_score_properties():
    y_true = np.array([1.0, 2.0, 4.0, 7.0])
    # Trajectories matching exact pairwise slope increments
    trajs_perfect = np.tile(y_true, (40, 1))
    vs_zero = variogram_score(y_true, trajs_perfect, p=0.5)
    assert math.isclose(vs_zero, 0.0, abs_tol=1e-8)

    # Inverted trajectories should produce high variogram penalty
    trajs_inverted = np.tile(y_true[::-1], (40, 1))
    vs_penalized = variogram_score(y_true, trajs_inverted, p=0.5)
    assert vs_penalized > 0.0


def test_continuous_winkler_score_monotonicity():
    y_true = 0.0
    narrow_samples = np.random.normal(0.0, 0.5, size=200)
    wide_samples = np.random.normal(0.0, 3.0, size=200)

    cw_narrow = continuous_winkler_score(y_true, narrow_samples)
    cw_wide = continuous_winkler_score(y_true, wide_samples)

    # When centered on truth, narrower interval yields better (lower) Winkler score
    assert cw_narrow < cw_wide


def test_pit_uniformity_and_anderson_darling():
    rng = np.random.default_rng(77)
    N = 100
    y_seq = []
    sample_seq = []

    # Well-calibrated sequence: observations drawn from N(0, 1) and predictions are N(0, 1)
    for _ in range(N):
        y = float(rng.normal(0.0, 1.0))
        samples = rng.normal(0.0, 1.0, size=150)
        y_seq.append(y)
        sample_seq.append(samples)

    pit_vals = compute_pit_values(y_seq, sample_seq)
    assert len(pit_vals) == N
    assert np.all(pit_vals >= 0.0) and np.all(pit_vals <= 1.0)

    diag = pit_anderson_darling_uniformity(pit_vals)
    # Calibrated distribution should pass uniformity at alpha=0.01
    assert diag.ks_pvalue > 0.01
    assert diag.anderson_statistic < 10.0


def test_calibration_and_sharpness_metrics():
    y_seq = np.linspace(-1.0, 1.0, 50)
    sample_seq = [np.random.normal(y, 0.5, size=80) for y in y_seq]

    report = evaluate_calibration_and_sharpness(y_seq, sample_seq)
    assert 0.0 <= report.expected_calibration_error <= 1.0
    assert 0.0 <= report.max_calibration_error <= 1.0
    assert report.mean_interval_width_90 > 0.0
    assert report.normalized_sharpness > 0.0


# ==============================================================================
# 3. Tests for Theory Bounds & Lyapunov Stability
# ==============================================================================

def test_dynamic_regret_bound_monotonicity():
    calc = DynamicRegretBoundCalculator(diameter_Z=4.0, gradient_bound_G=2.0)
    horizons = [1, 4, 8, 16]

    # Increasing path length P_T must monotonically increase the upper bound
    bound_low_p = calc.compute_bound(T=100, horizons=horizons, path_variation_P=1.0)
    bound_high_p = calc.compute_bound(T=100, horizons=horizons, path_variation_P=10.0)

    assert bound_high_p.total_dynamic_regret_bound > bound_low_p.total_dynamic_regret_bound
    assert bound_high_p.path_variation_penalty > bound_low_p.path_variation_penalty


def test_lyapunov_stability_certificate():
    cert = LyapunovStabilityCertificate.certify(
        latent_dim=2,
        learning_rate=0.08,
        step_norm_clip=0.30,
        damping_factor=0.95,
        max_gradient_norm=2.0,
    )
    assert cert.is_uniformly_bounded is True
    assert cert.spectral_radius == 0.95
    assert cert.compact_set_radius > 0.0
    # Compact ball radius R = max_step / (1 - rho) = 0.16 / 0.05 = 3.2
    assert cert.compact_set_radius == pytest.approx(0.16 / 0.05, rel=1e-3)


def test_empirical_bound_verifier_never_breaches_theoretical_bound():
    verifier = EmpiricalBoundVerifier(
        latent_dim=2,
        horizons=[1, 4],
        learning_rate=0.05,
        step_clip=0.30,
        seed=123,
    )
    points = verifier.run_verification_experiment(T=60, path_variations=(0.0, 2.0, 5.0))
    assert len(points) == 3
    for pt in points:
        # Crucial theoretical invariant: realized regret must be <= theoretical upper bound
        assert pt.bound_is_satisfied is True
        assert pt.empirical_bound_ratio <= 1.0


# ==============================================================================
# 4. Tests for Extended Benchmark Generators & Integration
# ==============================================================================

def test_markov_regime_switching_generator():
    gen = MarkovRegimeSwitchingGenerator(seed=42)
    series, states = gen.generate_series(100)
    assert len(series) == 100
    assert len(states) == 100
    # Must contain multiple regime states
    unique_states = np.unique(states)
    assert len(unique_states) >= 2


def test_heavy_tail_student_t_generator():
    gen = HeavyTailStudentTGenerator(degrees_of_freedom=2.5, seed=42)
    series = gen.generate_series(200)
    assert len(series) == 200
    # Student-t with df=2.5 exhibits extreme kurtosis / max absolute values
    assert np.max(np.abs(series)) > 2.5


def test_garch_volatility_generator():
    gen = GarchVolatilityClusterGenerator(omega=0.05, alpha=0.15, beta=0.80, seed=42)
    series = gen.generate_series(150)
    assert len(series) == 150
    assert np.all(np.isfinite(series))


def test_extended_benchmark_runner_integration():
    runner = ExtendedBenchmarkRunner(horizons=[1, 4], num_samples=30, seed=42)
    dummy_series = np.sin(np.linspace(0, 10, 50)) + np.random.normal(0, 0.2, 50)

    df = runner.run_benchmark_on_series(dummy_series, regime_name="TestSinusoid")
    assert len(df) == 10  # 4 baselines + 5 competitors + 1 proposed H3
    assert "Energy_Score" in df.columns
    assert "Variogram_05" in df.columns
    assert "Cont_Winkler" in df.columns
    assert "ECE" in df.columns
    assert "CRPS_Mean" in df.columns
    assert "CRPS_Rel_Imp_%" in df.columns
