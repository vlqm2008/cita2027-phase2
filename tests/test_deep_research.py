"""
Comprehensive Unit Tests for Advanced Frontier Research Modules:
1. Online Copula Trajectory Adapter (copula_adaptation.py)
2. Online Conformal Martingale Controller (conformal_martingale.py)
3. Natural Gradient Riemannian Adapter (natural_gradient.py)
4. Extreme Value Theory Tail Splicing Adapter (extreme_value_adaptation.py)
"""

import math
import numpy as np
import pytest

from cita2027.adaptation.conformal_martingale import OnlineConformalMartingaleController
from cita2027.adaptation.copula_adaptation import OnlineCopulaTrajectoryAdapter
from cita2027.adaptation.extreme_value_adaptation import OnlineExtremeValueAdapter
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.natural_gradient import NaturalGradientRiemannianAdapter
from cita2027.adaptation.probabilistic_metrics import energy_score


# ==============================================================================
# 1. Tests for Copula Adaptation
# ==============================================================================

def test_copula_adapter_covariance_positive_definite():
    horizons = [1, 4, 8, 16]
    copula = OnlineCopulaTrajectoryAdapter(horizons, num_samples=50)

    # Initial covariance matrix must be symmetric positive definite
    assert np.allclose(copula.Sigma, copula.Sigma.T)
    eigvals = np.linalg.eigvals(copula.Sigma)
    assert np.all(eigvals > 0)

    # Update with synthetic correlated residuals
    r = np.array([1.2, 1.0, 0.8, 0.5])
    copula.update_covariance(r)

    # Must remain symmetric positive definite
    assert np.allclose(copula.Sigma, copula.Sigma.T)
    eigvals_after = np.linalg.eigvals(copula.Sigma)
    assert np.all(eigvals_after > 0)


def test_copula_trajectory_warping_preserves_marginals():
    horizons = [1, 4]
    copula = OnlineCopulaTrajectoryAdapter(horizons, num_samples=100)

    marginals = {
        1: np.random.normal(5.0, 1.0, size=100),
        4: np.random.normal(10.0, 2.0, size=100),
    }

    trajs = copula.warp_trajectories(marginals)
    assert trajs.shape == (100, 2)

    # Marginal medians should match original samples closely
    assert math.isclose(float(np.median(trajs[:, 0])), float(np.median(marginals[1])), abs_tol=0.2)
    assert math.isclose(float(np.median(trajs[:, 1])), float(np.median(marginals[4])), abs_tol=0.3)


# ==============================================================================
# 2. Tests for Conformal Martingale Controller
# ==============================================================================

def test_conformal_martingale_wealth_and_rejection():
    horizons = [1, 4]
    controller = OnlineConformalMartingaleController(
        horizons=horizons,
        nominal_coverage=0.90,
        significance_level_delta=0.10,  # threshold = 10
    )

    base_s = np.random.normal(0, 1, size=100)

    # Feed repeated extreme miscoverage observations (y = 50.0)
    for step in range(15):
        state = controller.evaluate_and_update(horizon=1, y_true=50.0, pred_samples=base_s, step=step)

    # Wealth process must grow and trigger rejection
    assert state.current_wealth > 1.0
    assert state.is_calibration_rejected is True

    # Calibrate samples should expand predictive intervals
    cal_s = controller.calibrate_samples(1, base_s)
    assert np.std(cal_s) > np.std(base_s)


def test_conformal_martingale_calibrated_behavior():
    horizons = [1]
    controller = OnlineConformalMartingaleController(
        horizons=horizons,
        nominal_coverage=0.90,
        significance_level_delta=0.01,
    )

    rng = np.random.default_rng(42)
    # Feed well-calibrated draws from N(0, 1)
    for step in range(30):
        samples = rng.normal(0, 1, size=100)
        y = float(rng.normal(0, 1))
        state = controller.evaluate_and_update(1, y, samples, step)

    # Calibration should not be rejected
    assert state.is_calibration_rejected is False
    assert state.current_wealth < 100.0


# ==============================================================================
# 3. Tests for Natural Gradient Riemannian Adapter
# ==============================================================================

def test_natural_gradient_adapter_fisher_matrix():
    horizons = [1, 4, 8]
    adapter = NaturalGradientRiemannianAdapter(horizons, latent_dim=2, step_clip=0.35)

    assert adapter.FIM_loc.shape == (2, 2)
    assert np.all(np.linalg.eigvals(adapter.FIM_loc) > 0)

    base_s = np.random.normal(0, 1, size=40)
    fb = [
        ForecastFeedbackItem(
            origin_t=0,
            horizon=1,
            target_t=1,
            y_true=3.0,
            base_samples=base_s,
            adapted_samples=base_s,
        )
    ]
    adapter.update(fb)

    # State vectors should be updated
    assert np.linalg.norm(adapter.z) > 0.0
    assert np.linalg.norm(adapter.z) <= 0.35 + 1e-6

    # Prediction should shift upward
    pred = adapter.adapt(1, 1, base_s)
    assert np.median(pred) > np.median(base_s)


def test_natural_gradient_boundedness():
    adapter = NaturalGradientRiemannianAdapter([1, 4], latent_dim=2, step_clip=0.20)
    base_s = np.zeros(30)

    # Massive residual shock
    fb = [ForecastFeedbackItem(0, 1, 1, 1000.0, base_s, base_s)]
    adapter.update(fb)

    # Norm must not exceed step_clip
    assert np.linalg.norm(adapter.z) <= 0.20 + 1e-6


# ==============================================================================
# 4. Tests for Extreme Value Theory Tail Splicing
# ==============================================================================

def test_extreme_value_tail_splicing_properties():
    horizons = [1, 4]
    evt = OnlineExtremeValueAdapter(horizons, tail_probability=0.10)

    base_s = np.random.normal(0, 1, size=100)

    # Feed extreme shocks to horizon 1
    for shock in [4.5, 5.2, 6.0, 7.5, 8.0, 6.8]:
        evt.update_tails(1, y_true=shock, base_median=0.0, base_std=1.0)

    # Spliced samples
    spliced = evt.splice_predictive_samples(1, base_s)
    assert len(spliced) == 100
    # Must remain monotonically sorted
    assert np.all(np.diff(spliced) >= 0)

    # Central median should remain virtually unchanged
    assert math.isclose(float(np.median(spliced)), float(np.median(base_s)), abs_tol=0.1)

    # Extreme upper tail (99th percentile) should be higher than normal draws
    assert np.quantile(spliced, 0.99) > np.quantile(base_s, 0.99)
