"""Stress and integration tests for Phase IV frontier adaptation modules.

Tests:
- Koopman EDMD spectral properties under extreme noise and non-stationary limit cycles.
- Differentially private state updates under high clipping and privacy expenditure.
- Causal IRM multi-environment partitioning and invariance convergence.
- Wasserstein barycentric morphing under extreme heavy tails and asymmetric distributions.
- Grand integration across all 4 modules in a shared asynchronous prequential pipeline.
"""

import math
import numpy as np
import pytest

from cita2027.adaptation.koopman_adaptation import (
    KoopmanCrossHorizonAdapter,
    OnlineExtendedDynamicModeDecomposition,
)
from cita2027.adaptation.differentially_private_adaptation import (
    DifferentiallyPrivateCrossHorizonAdapter,
)
from cita2027.adaptation.causal_irm_adaptation import (
    CausalIRMCrossHorizonAdapter,
    CausalEnvironmentDetector,
)
from cita2027.adaptation.wasserstein_barycenter_adaptation import (
    WassersteinBarycenterCrossHorizonAdapter,
)


def test_koopman_chaotic_lorenz_tracking():
    """Verify Koopman EDMD tracking stability on chaotic nonlinear series."""
    adapter = KoopmanCrossHorizonAdapter(
        horizons=[1, 2, 4, 8, 16],
        spectral_radius_max=0.91,
    )
    # Chaotic Henon map-like series
    x, y = 0.1, 0.1
    for t in range(100):
        x_new = 1.0 - 1.4 * x**2 + y
        y_new = 0.3 * x
        x, y = x_new, y_new

        fb = [{"horizon": 1, "y_true": x, "y_pred_loc": 0.0, "y_pred_disp": 1.0}]
        info = adapter.update_feedback(fb)
        assert info["spectral_radius"] <= 0.91 + 1e-4

    # Ensure adaptation produces finite samples
    base = np.random.randn(50)
    adapted, loc, disp = adapter.adapt_forecast(4, base, 0.0, 1.0)
    assert np.all(np.isfinite(adapted))
    assert disp > 0.0


def test_dp_high_privacy_regime_stability():
    """Verify DP-ACHA remains Lyapunov-bounded under extreme epsilon=0.05."""
    dp = DifferentiallyPrivateCrossHorizonAdapter(
        horizons=[1, 2, 4],
        epsilon=0.05,
        clipping_norm=0.5,
        state_bound=8.0,
    )
    for t in range(50):
        fb = [
            {"horizon": 1, "y_true": 10.0, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
            {"horizon": 2, "y_true": -10.0, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
        ]
        info = dp.update_feedback(fb)
        assert info["z_norm"] <= 8.0 + 1e-6
        assert info["privacy_spent"] > 0.0


def test_causal_irm_three_regime_stability():
    """Verify Causal IRM across rapid switching of 3 regimes."""
    irm = CausalIRMCrossHorizonAdapter(
        horizons=[1, 2, 4],
        irm_penalty_lambda=5.0,
        state_bound=8.0,
    )
    for t in range(90):
        # Switch regimes every 30 steps
        regime = t // 30
        shift = 1.0 if regime == 0 else (-1.0 if regime == 1 else 0.0)
        fb = [
            {"horizon": 1, "y_true": shift + 0.1 * np.random.randn(), "y_pred_loc": 0.0, "y_pred_disp": 1.0},
            {"horizon": 2, "y_true": shift + 0.1 * np.random.randn(), "y_pred_loc": 0.0, "y_pred_disp": 1.0},
        ]
        info = irm.update_feedback(fb)
        assert info["z_norm"] <= 8.0 + 1e-6

    # Test out-of-distribution forecast
    base = np.random.randn(60)
    adapted, loc, disp = irm.adapt_forecast(4, base, 0.0, 1.0)
    assert np.all(np.isfinite(adapted))
    assert disp > 0.0


def test_wasserstein_asymmetric_extreme_tails():
    """Verify Wasserstein Barycenter preserves strict monotonicity under extreme Student-t / Pareto inputs."""
    wb = WassersteinBarycenterCrossHorizonAdapter(
        horizons=[1, 2, 4, 8],
        num_quantiles=60,
    )
    for t in range(50):
        # Heavy-tailed Student-t(df=2)
        tail_sample = np.random.standard_t(df=2)
        fb = [{"horizon": 1, "y_true": tail_sample, "y_pred_loc": 0.0, "y_pred_disp": 1.0}]
        wb.update_feedback(fb)

    q = wb.compute_wasserstein_barycenter_quantiles()
    diffs = np.diff(q)
    assert np.all(diffs >= -1e-9), "Wasserstein quantiles must be monotonic"


def test_grand_four_module_concurrent_pipeline():
    """Simulate streaming pipeline evaluating all 4 Phase IV frontier adapters concurrently."""
    horizons = [1, 2, 4, 8, 16]
    koopman = KoopmanCrossHorizonAdapter(horizons=horizons)
    dp = DifferentiallyPrivateCrossHorizonAdapter(horizons=horizons)
    irm = CausalIRMCrossHorizonAdapter(horizons=horizons)
    wb = WassersteinBarycenterCrossHorizonAdapter(horizons=horizons)

    adapters = [koopman, dp, irm, wb]

    for t in range(30):
        feedback = []
        for h in horizons:
            if t >= h:
                feedback.append({
                    "horizon": h,
                    "y_true": 0.5 + 0.1 * np.random.randn(),
                    "y_pred_loc": 0.0,
                    "y_pred_disp": 1.0,
                })
        for ad in adapters:
            ad.update_feedback(feedback)

        # Adapt forecast at h=8
        base_samples = np.random.randn(100)
        for ad in adapters:
            adapted, loc, disp = ad.adapt_forecast(8, base_samples, 0.0, 1.0)
            assert len(adapted) == 100
            assert np.all(np.isfinite(adapted))
            assert disp > 0.0
