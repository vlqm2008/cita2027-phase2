"""
Stress and Edge-Case Robustness Tests for Advanced Frontiers Phase II.
"""

import numpy as np
import pytest

from cita2027.adaptation.changepoint_martingale import (
    ChangePointMartingaleAdapter,
    StreamingChangePointMartingale,
)
from cita2027.adaptation.extreme_conformal_tails import (
    ExtremeValueConformalAdapter,
    OnlineExtremeConformalCalibrator,
)
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.multifidelity_adaptation import (
    MultiFidelityCrossHorizonAdapter,
    MultiFidelityFeedbackItem,
)
from cita2027.adaptation.riemannian_natural_gradient import (
    RiemannianNaturalGradientAdapter,
)


def test_changepoint_martingale_zero_variance_protection():
    cpm = StreamingChangePointMartingale(threshold_b=10.0, burnin_steps=5)
    # Feed identical values (zero empirical variance)
    for _ in range(15):
        m_val, is_det = cpm.update(1.0)
        assert not is_det
        assert not np.isnan(m_val)


def test_riemannian_near_singular_damping():
    horizons = [1, 2]
    adapter = RiemannianNaturalGradientAdapter(horizons=horizons, latent_dim=2, fisher_reg=1e-6)
    # Deliberately set scales to cause high condition numbers
    g_mat = adapter.compute_fisher_information_matrix()
    assert not np.isnan(g_mat).any()

    # Step with zero residual
    items = [
        ForecastFeedbackItem(
            origin_t=0, horizon=1, target_t=1, y_true=0.0, base_samples=np.zeros(10)
        )
    ]
    res = adapter.update_with_feedback(items, current_t=1)
    assert not np.isnan(res["fisher_cond"])
    assert res["state_norm"] <= 8.0


def test_multifidelity_extreme_variance_ratio():
    horizons = [1, 4]
    # Extreme variance ratio: proxy variance = 1e6, verified = 1e-4
    adapter = MultiFidelityCrossHorizonAdapter(
        horizons=horizons, low_fidelity_variance=1e6, high_fidelity_variance=1e-4
    )
    proxy_items = [
        MultiFidelityFeedbackItem(
            origin_t=0, horizon=1, target_t=0, y_true=100.0, base_samples=np.zeros(10), fidelity_tier="low"
        )
    ]
    # Noisy proxy with massive variance should have negligible effect
    res = adapter.update_with_feedback(proxy_items, current_t=0)
    assert res["state_norm"] < 0.1


def test_extreme_cps_insufficient_samples_fallback():
    calibrator = OnlineExtremeConformalCalibrator(nominal_coverage=0.99, buffer_size=100)
    # Under small sample count (< 20), must return valid conservative default
    for s in [0.5, 1.2, 0.8]:
        q_val = calibrator.observe_score(s)
        assert 0.5 <= q_val <= 15.0


def test_end_to_end_cpm_riemann_synergy():
    horizons = [1, 2, 4]
    cpm = ChangePointMartingaleAdapter(horizons=horizons, latent_dim=2)
    riemann = RiemannianNaturalGradientAdapter(horizons=horizons, latent_dim=2)

    for t in range(10):
        items = [
            ForecastFeedbackItem(
                origin_t=max(0, t - h), horizon=h, target_t=t, y_true=1.5, base_samples=np.full(10, 1.0)
            )
            for h in horizons
        ]
        res_c = cpm.update_with_feedback(items, current_t=t)
        res_r = riemann.update_with_feedback(items, current_t=t)
        assert res_c["state_norm"] <= 8.0
        assert res_r["state_norm"] <= 8.0
