"""
Unit Tests for Advanced Frontiers Phase II:
- Streaming Change-Point Martingale (CPM-H3)
- Information-Geometric Riemannian Natural Gradient (Riemann-H3)
- Asynchronous Multi-Fidelity Cross-Horizon Adaptation (MultiFidelity-H3)
- Extreme-Value Conformal Predictive Set Calibration (Extreme-CPS)
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


def test_changepoint_martingale_detector():
    cpm = StreamingChangePointMartingale(threshold_b=15.0, burnin_steps=10)

    # 1. Burn-in period with stationary loss
    for _ in range(12):
        m_val, is_detected = cpm.update(0.5)
        assert not is_detected

    # 2. Large abrupt structural shift in loss
    detected = False
    for _ in range(10):
        m_val, is_det = cpm.update(5.0)
        if is_det:
            detected = True
            break

    assert detected, "Change-point martingale should detect sudden 10x loss surge"


def test_changepoint_adapter_burst_mode():
    horizons = [1, 2, 4]
    adapter = ChangePointMartingaleAdapter(
        horizons=horizons,
        base_lr=0.05,
        burst_multiplier=3.0,
        cpm_threshold=10.0,
    )

    # Feed large errors to trigger burst adaptation
    for t in range(15):
        items = [
            ForecastFeedbackItem(
                origin_t=max(0, t - h),
                horizon=h,
                target_t=t,
                y_true=10.0,
                base_samples=np.full(10, 0.0),
            )
            for h in horizons
        ]
        res = adapter.update_with_feedback(items, current_t=t)
        assert res["state_norm"] <= 8.0

    assert adapter.current_lr >= 0.05
    loc, scale = adapter.get_corrections(2)
    assert scale > 0.0


def test_riemannian_natural_gradient_fisher_metric():
    horizons = [1, 2, 4, 8]
    adapter = RiemannianNaturalGradientAdapter(horizons=horizons, latent_dim=2, fisher_reg=0.1)

    # Fisher Information Matrix should be symmetric positive-definite
    g_mat = adapter.compute_fisher_information_matrix()
    assert g_mat.shape == (2, 2)
    assert np.allclose(g_mat, g_mat.T)
    eigvals = np.linalg.eigvalsh(g_mat)
    assert np.all(eigvals > 0.0)

    # Ingest feedback and run natural gradient step
    items = [
        ForecastFeedbackItem(
            origin_t=0,
            horizon=h,
            target_t=h,
            y_true=2.5,
            base_samples=np.full(15, 1.0),
        )
        for h in horizons
    ]
    res = adapter.update_with_feedback(items, current_t=8)
    assert res["fisher_cond"] >= 1.0
    assert res["state_norm"] <= 8.0

    samples = adapter.get_adapted_samples(4, np.array([1.0, 2.0, 3.0]))
    assert len(samples) == 3
    assert not np.isnan(samples).any()


def test_multifidelity_adaptation_fusion():
    horizons = [1, 2, 4]
    adapter = MultiFidelityCrossHorizonAdapter(horizons=horizons)

    # Step 1: Ingest low-fidelity proxy feedback at t=1
    proxy_items = [
        MultiFidelityFeedbackItem(
            origin_t=1,
            horizon=h,
            target_t=1,
            y_true=3.0,
            base_samples=np.full(10, 1.0),
            fidelity_tier="low",
            noise_variance=1.0,
        )
        for h in horizons
    ]
    res1 = adapter.update_with_feedback(proxy_items, current_t=1)
    assert res1["low_fidelity_count"] == 3
    assert (1, 1) in adapter.proxy_cache

    # Step 2: High-fidelity verification arrives at t=5 for origin t=1
    verified_items = [
        MultiFidelityFeedbackItem(
            origin_t=1,
            horizon=h,
            target_t=5,
            y_true=3.2,
            base_samples=np.full(10, 1.0),
            fidelity_tier="high",
            noise_variance=0.05,
        )
        for h in horizons
    ]
    res2 = adapter.update_with_feedback(verified_items, current_t=5)
    assert res2["high_fidelity_count"] == 3
    # Proxy cache should have been popped/consumed
    assert (1, 1) not in adapter.proxy_cache
    assert res2["state_norm"] <= 8.0


def test_extreme_conformal_calibrator_and_adapter():
    horizons = [1, 4]
    adapter = ExtremeValueConformalAdapter(horizons=horizons, nominal_coverage=0.95)

    # Ingest synthetic heavy-tailed Student-t(3) residuals
    rng = np.random.RandomState(42)
    for t in range(30):
        items = [
            ForecastFeedbackItem(
                origin_t=0,
                horizon=h,
                target_t=t,
                y_true=float(rng.standard_t(df=3)),
                base_samples=rng.normal(0, 1, size=15),
            )
            for h in horizons
        ]
        res = adapter.update_with_feedback(items, current_t=t)

    assert res["mean_extreme_conformal_q"] >= 1.0

    # Test prediction interval
    low, high = adapter.get_prediction_interval(horizon=4, base_location=0.0, base_dispersion=1.0)
    assert low < high
    assert high >= 1.0
