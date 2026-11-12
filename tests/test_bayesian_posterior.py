"""
Unit Tests for Bayesian Online Posterior Adaptation (Bayes-H3).
"""

import numpy as np
import pytest

from cita2027.adaptation.bayesian_online_posterior import (
    BayesianCrossHorizonPosteriorTracker,
)
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


def test_bayesian_posterior_covariance_contraction():
    horizons = [1, 2, 4, 8]
    tracker = BayesianCrossHorizonPosteriorTracker(
        horizons=horizons,
        latent_dim=2,
        prior_variance=2.0,
        obs_noise_r=0.1,
    )

    initial_trace = np.trace(tracker.cov_z)
    assert initial_trace == pytest.approx(4.0)

    # Ingest informative feedback
    for t in range(1, 10):
        items = [
            ForecastFeedbackItem(
                origin_t=0,
                horizon=h,
                target_t=t,
                y_true=1.5,
                base_samples=np.full(10, 1.0),
            )
            for h in horizons
        ]
        tracker.update_with_feedback(items, current_t=t)

    final_trace = np.trace(tracker.cov_z)
    # Variance must contract with evidence
    assert final_trace < initial_trace
    # Covariance must remain symmetric and positive-definite
    assert np.allclose(tracker.cov_z, tracker.cov_z.T)
    eigvals = np.linalg.eigvalsh(tracker.cov_z)
    assert np.all(eigvals > 0.0)


def test_bayesian_epistemic_hedging_and_scale():
    horizons = [1, 4]
    tracker = BayesianCrossHorizonPosteriorTracker(horizons=horizons, latent_dim=2)

    loc, scale = tracker.get_corrections(4)
    assert scale > 0.0
    assert 0.1 <= scale <= 10.0

    samples = np.array([5.0, 6.0, 7.0])
    adapted = tracker.get_adapted_samples(horizon=4, base_samples=samples)
    assert len(adapted) == len(samples)
    assert not np.isnan(adapted).any()


def test_bayesian_posterior_boundedness():
    horizons = [1, 2]
    tracker = BayesianCrossHorizonPosteriorTracker(horizons=horizons, latent_dim=2)

    # Extreme observations to test norm clamping
    items = [
        ForecastFeedbackItem(
            origin_t=0,
            horizon=1,
            target_t=1,
            y_true=1000.0,
            base_samples=np.zeros(10),
        )
    ]
    tracker.update_with_feedback(items, current_t=1)
    assert np.linalg.norm(tracker.mean_z) <= 8.0 + 1e-6
