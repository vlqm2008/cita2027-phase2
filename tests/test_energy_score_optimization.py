"""
Unit Tests for Direct Multivariate Energy Score Online Optimization (ES-H3).
"""

import numpy as np
import pytest

from cita2027.adaptation.energy_score_optimization import HypothesisESH3_DirectOptimizer
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


def test_es_h3_corrections_and_adaptation():
    horizons = [1, 2, 4, 8, 16]
    optimizer = HypothesisESH3_DirectOptimizer(horizons=horizons, latent_dim=2)

    delta_mu, scale_mult = optimizer.get_corrections(8)
    assert delta_mu == 0.0  # Initially zero
    assert scale_mult == 1.0  # Initially exp(0) = 1.0

    samples = np.random.normal(5.0, 1.0, size=50)
    adapted = optimizer.adapt_samples(samples, base_location=5.0, h=8)
    assert len(adapted) == 50
    assert np.all(np.isfinite(adapted))


def test_es_h3_subgradient_update_reduces_loss():
    horizons = [1, 2, 4]
    optimizer = HypothesisESH3_DirectOptimizer(horizons=horizons, latent_dim=2, learning_rate=0.1)

    # Systematic positive shift
    true_target = 12.0
    initial_err = abs(5.0 - true_target)

    for t in range(20):
        items = [
            ForecastFeedbackItem(
                origin_t=t,
                horizon=h,
                target_t=t + h,
                y_true=true_target,
                base_samples=np.ones(30) * 5.0,
            )
            for h in horizons
        ]
        info = optimizer.update_with_feedback(items, current_t=t + 4)
        assert info["z_norm"] <= 8.0

    # Adapted location should move toward true_target
    delta_mu, _ = optimizer.get_corrections(1)
    final_err = abs(5.0 + delta_mu - true_target)
    assert final_err < initial_err, "Subgradient steps must reduce discrepancy toward target"
