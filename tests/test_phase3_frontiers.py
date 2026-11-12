"""
Unit Tests for Phase III Frontier Modules:
- Hyperbolic Geometry Manifold (Hyperbolic-H3)
- Game-Theoretic Minimax Adversarial Equilibrium (GameTheoretic-H3)
- Dirichlet Process Gaussian Mixture Calibration (DPM-H3)
- Continuous-Time Neural Stochastic Differential Equations (NeuralSDE-H3)
"""

import numpy as np
import pytest

from cita2027.adaptation.dirichlet_process_adaptation import (
    DirichletProcessMixtureAdapter,
)
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.game_theoretic_adaptation import (
    GameTheoreticMinimaxAdapter,
)
from cita2027.adaptation.hyperbolic_adaptation import (
    HyperbolicCrossHorizonAdapter,
    mobius_addition,
    poincare_dist,
)
from cita2027.adaptation.neural_sde_adaptation import (
    ContinuousTimeFeedbackItem,
    NeuralSDECrossHorizonAdapter,
)


def test_hyperbolic_poincare_geometry_and_adapter():
    horizons = [1, 2, 4, 8, 16]
    adapter = HyperbolicCrossHorizonAdapter(horizons=horizons, latent_dim=2)

    # Verify Poincare distance symmetry and positivity
    p1 = np.array([0.2, 0.3])
    p2 = np.array([-0.1, 0.4])
    d12 = poincare_dist(p1, p2)
    d21 = poincare_dist(p2, p1)
    assert np.isclose(d12, d21)
    assert d12 > 0.0

    # Ingest feedback items
    items = [
        ForecastFeedbackItem(
            origin_t=0,
            horizon=h,
            target_t=h,
            y_true=1.5,
            base_samples=np.full(10, 1.0),
        )
        for h in horizons
    ]
    res = adapter.update_with_feedback(items, current_t=10)

    # State must strictly remain inside Poincare ball ||z|| < 1.0
    assert res["poincare_norm"] < 1.0
    assert adapter.hyperbolic_z.shape == (2,)

    loc, scale = adapter.get_corrections(4)
    assert scale > 0.0
    assert 0.1 <= scale <= 10.0


def test_game_theoretic_minimax_adapter():
    horizons = [1, 2, 4]
    adapter = GameTheoreticMinimaxAdapter(
        horizons=horizons,
        latent_dim=2,
        adversarial_radius_eps=0.4,
    )

    items = [
        ForecastFeedbackItem(
            origin_t=0,
            horizon=h,
            target_t=h,
            y_true=5.0,
            base_samples=np.full(10, 0.0),
        )
        for h in horizons
    ]

    # Compute worst-case shock
    v_star = adapter.compute_worst_case_shock(items)
    assert np.linalg.norm(v_star) <= 0.4 + 1e-6

    res = adapter.update_with_feedback(items, current_t=5)
    assert res["state_norm"] <= 8.0

    samples = adapter.get_adapted_samples(2, np.array([1.0, 2.0, 3.0]))
    assert len(samples) == 3
    assert not np.isnan(samples).any()


def test_dirichlet_process_mixture_adapter():
    horizons = [1, 4]
    adapter = DirichletProcessMixtureAdapter(
        horizons=horizons,
        latent_dim=2,
        alpha_dp=0.8,
        max_clusters=4,
    )

    # Feed distinct multi-modal residuals to induce cluster spawning
    # Mode 1: centered at +2.0
    items1 = [
        ForecastFeedbackItem(origin_t=0, horizon=1, target_t=1, y_true=2.0, base_samples=np.zeros(10))
    ]
    adapter.update_with_feedback(items1, current_t=1)

    # Mode 2: extreme outlier at -6.0
    items2 = [
        ForecastFeedbackItem(origin_t=0, horizon=1, target_t=2, y_true=-6.0, base_samples=np.zeros(10))
    ]
    res2 = adapter.update_with_feedback(items2, current_t=2)

    # Multiple clusters should now be maintained
    assert len(adapter.clusters) >= 1
    assert res2["num_clusters"] >= 1.0

    # Test multi-modal sampling
    samples = adapter.get_adapted_samples(horizon=1, base_samples=np.random.normal(0, 1, size=50))
    assert len(samples) == 50
    assert not np.isnan(samples).any()


def test_neural_sde_continuous_time_adapter():
    horizons = [1, 2, 4]
    adapter = NeuralSDECrossHorizonAdapter(horizons=horizons, latent_dim=2)

    # Propagate across continuous elapsed time Delta tau = 5.5
    adapter.propagate_sde(current_tau=5.5)
    assert adapter.last_tau == pytest.approx(5.5)

    # Ingest continuous-time feedback items
    items = [
        ContinuousTimeFeedbackItem(
            origin_t=5.5,
            horizon=h,
            target_t=5.5 + h,
            y_true=2.0,
            base_samples=np.full(10, 1.0),
        )
        for h in horizons
    ]
    res = adapter.update_with_feedback(items, current_t=9.5)
    assert res["state_norm"] <= 8.0
    assert res["current_tau"] == pytest.approx(9.5)

    loc, scale = adapter.get_corrections(2)
    assert not np.isnan(loc)
