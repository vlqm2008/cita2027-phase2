"""
Comprehensive Synthesis & Stress Tests for Phase III Frontier Paradigms.
"""

import numpy as np
import pytest

from cita2027.adaptation.changepoint_martingale import (
    ChangePointMartingaleAdapter,
)
from cita2027.adaptation.dirichlet_process_adaptation import (
    DirichletProcessMixtureAdapter,
)
from cita2027.adaptation.extreme_conformal_tails import (
    ExtremeValueConformalAdapter,
)
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.game_theoretic_adaptation import (
    GameTheoreticMinimaxAdapter,
)
from cita2027.adaptation.hyperbolic_adaptation import (
    HyperbolicCrossHorizonAdapter,
    poincare_dist,
)
from cita2027.adaptation.metrics import crps_empirical_samples
from cita2027.adaptation.neural_sde_adaptation import (
    ContinuousTimeFeedbackItem,
    NeuralSDECrossHorizonAdapter,
)


def test_hyperbolic_ultralong_horizon_scaling():
    ultralong_horizons = [1, 2, 4, 8, 16, 32, 64]
    adapter = HyperbolicCrossHorizonAdapter(horizons=ultralong_horizons, latent_dim=2)

    # Verify that all embeddings reside strictly inside Poincare disk
    for h, u in adapter.horizon_embeddings.items():
        assert np.linalg.norm(u) < 1.0

    # Distance between consecutive dyadic horizons should remain bounded O(1)
    dists = []
    for i in range(len(ultralong_horizons) - 1):
        h1 = ultralong_horizons[i]
        h2 = ultralong_horizons[i + 1]
        d = poincare_dist(adapter.horizon_embeddings[h1], adapter.horizon_embeddings[h2])
        dists.append(d)

    assert max(dists) < 5.0, "Hyperbolic dyadic embedding distance must remain bounded O(1)"


def test_adversarial_game_robustness_bound():
    horizons = [1, 2, 4]
    adapter = GameTheoreticMinimaxAdapter(horizons=horizons, adversarial_radius_eps=0.5)

    # Ingest series of adversarially directed noise items
    for t in range(1, 15):
        items = [
            ForecastFeedbackItem(
                origin_t=0, horizon=h, target_t=t, y_true=np.sin(t) * 2.0, base_samples=np.zeros(10)
            )
            for h in horizons
        ]
        res = adapter.update_with_feedback(items, current_t=t)
        assert res["state_norm"] <= 8.0


def test_dirichlet_multi_modal_crps_propriety():
    horizons = [1, 4]
    adapter = DirichletProcessMixtureAdapter(horizons=horizons, alpha_dp=0.6)

    # Create two clusters
    items = [
        ForecastFeedbackItem(origin_t=0, horizon=1, target_t=1, y_true=3.0, base_samples=np.zeros(10)),
        ForecastFeedbackItem(origin_t=0, horizon=1, target_t=2, y_true=-3.0, base_samples=np.zeros(10)),
    ]
    adapter.update_with_feedback(items, current_t=2)

    samples = adapter.get_adapted_samples(horizon=1, base_samples=np.random.normal(0, 1, size=40))
    crps_val = crps_empirical_samples(samples, 0.0)
    assert not np.isnan(crps_val)
    assert crps_val > 0.0


def test_continuous_time_irregular_sampling_jitter():
    horizons = [1, 2]
    adapter = NeuralSDECrossHorizonAdapter(horizons=horizons, latent_dim=2)

    # Random exponential inter-arrival times
    current_time = 0.0
    rng = np.random.RandomState(42)
    for _ in range(15):
        delta_tau = float(rng.exponential(scale=1.5))
        current_time += delta_tau
        items = [
            ContinuousTimeFeedbackItem(
                origin_t=current_time - 1.0,
                horizon=1,
                target_t=current_time,
                y_true=float(np.cos(current_time)),
                base_samples=np.zeros(10),
            )
        ]
        res = adapter.update_with_feedback(items, current_t=current_time)
        assert res["state_norm"] <= 8.0


def test_grand_cross_paradigm_composition():
    horizons = [1, 4]
    hyp = HyperbolicCrossHorizonAdapter(horizons=horizons)
    cpm = ChangePointMartingaleAdapter(horizons=horizons)
    evt = ExtremeValueConformalAdapter(horizons=horizons, nominal_coverage=0.95)

    base_s = np.random.normal(0, 1, size=30)
    items = [
        ForecastFeedbackItem(origin_t=0, horizon=4, target_t=4, y_true=2.5, base_samples=base_s)
    ]

    res_h = hyp.update_with_feedback(items, current_t=4)
    res_c = cpm.update_with_feedback(items, current_t=4)
    res_e = evt.update_with_feedback(items, current_t=4)

    samples_h = hyp.get_adapted_samples(4, base_s)
    samples_c = cpm.get_adapted_samples(4, base_s)
    low, high = evt.get_prediction_interval(4, 0.0, 1.0)

    assert not np.isnan(samples_h).any()
    assert not np.isnan(samples_c).any()
    assert low < high
