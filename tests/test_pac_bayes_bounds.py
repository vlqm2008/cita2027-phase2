"""Unit tests for Theorem 2 PAC-Bayesian Cross-Horizon Generalization Bounds."""
from __future__ import annotations

import math
import numpy as np
import pytest

from cita2027.adaptation.pac_bayes_bounds import (
    CrossHorizonPACBayesEvaluator,
    PACBayesBoundResult,
)


def test_pac_bayes_evaluator_initialization():
    horizons = [1, 4, 8, 16]
    evaluator = CrossHorizonPACBayesEvaluator(horizons=horizons, latent_dim=2, delta=0.05)
    assert evaluator.horizons == [1, 4, 8, 16]
    assert evaluator.latent_dim == 2
    assert evaluator.delta == 0.05


def test_manifold_kl_divergence_non_negative():
    evaluator = CrossHorizonPACBayesEvaluator(horizons=[1, 4], latent_dim=2)
    np.random.seed(42)
    states = np.random.randn(100, 2) * 0.5 + 0.2
    kl = evaluator.compute_manifold_kl_divergence(states)
    assert kl >= 0.0
    assert not math.isnan(kl)
    assert not math.isinf(kl)


def test_total_variation_drift_monotonicity():
    evaluator = CrossHorizonPACBayesEvaluator(horizons=[1, 4])
    # Stationary series
    stationary_obs = np.random.randn(200)
    # Severe drifting series
    drifting_obs = np.sin(np.linspace(0, 10, 200)) * 5.0 + np.random.randn(200)

    tv_stat = evaluator.estimate_total_variation_drift(stationary_obs, window=20)
    tv_drift = evaluator.estimate_total_variation_drift(drifting_obs, window=20)
    assert tv_drift > tv_stat


def test_overlap_spectral_norm_properties():
    horizons = [1, 4, 8, 16]
    evaluator = CrossHorizonPACBayesEvaluator(horizons=horizons)
    latencies = {1: 1, 4: 4, 8: 8, 16: 16}
    l_max = evaluator.compute_overlap_spectral_norm(4, latencies)
    assert l_max >= 1.0  # Normalized overlap matrix trace >= K, max eigenvalue >= 1.0
    assert not math.isnan(l_max)


def test_theorem2_bound_holds_on_empirical_data():
    horizons = [1, 4, 8, 16]
    evaluator = CrossHorizonPACBayesEvaluator(horizons=horizons, latent_dim=2, delta=0.05)
    
    np.random.seed(2027)
    T = 200
    obs = np.cumsum(np.random.randn(T) * 0.1)
    states = np.random.randn(T, 2) * 0.1

    # Horizon-dependent comparator risk reflects multi-horizon dispersion growth
    oracle_crps = {1: 0.55, 4: 0.78, 8: 0.88, 16: 0.98}
    empirical_crps = {1: 0.62, 4: 0.82, 8: 0.92, 16: 1.02}
    suite_results = evaluator.evaluate_suite(
        empirical_crps_by_horizon=empirical_crps,
        shared_state_history=states,
        observations=obs,
        oracle_crps=oracle_crps,
    )

    for h, res in suite_results.items():
        assert isinstance(res, PACBayesBoundResult)
        assert res.horizon == h
        assert res.time_horizon_T == T
        assert res.theoretical_upper_bound > 0
        assert res.complexity_penalty > 0
        assert res.drift_penalty >= 0
        assert res.overlap_penalty > 0
        # Check that bound is valid
        assert res.bound_holds, f"Theorem 2 violated at horizon {h}: {res}"
