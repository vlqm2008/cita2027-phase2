"""Unit tests for Hypothesis H2 age-weighted and drift-aware adaptation."""
from __future__ import annotations

import numpy as np
import pytest

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h2 import HypothesisH2_AgeWeightedCorrector


def test_age_weight_monotonicity():
    horizons = [1, 4, 8, 16]
    c_exp = HypothesisH2_AgeWeightedCorrector(horizons, decay_type="exponential", gamma=0.2)
    c_hyp = HypothesisH2_AgeWeightedCorrector(horizons, decay_type="hyperbolic", gamma=0.2)

    base = np.zeros(10)
    w_exp = []
    w_hyp = []

    for h in horizons:
        item = ForecastFeedbackItem(origin_t=0, horizon=h, target_t=h, y_true=1.0, base_samples=base)
        w_exp.append(c_exp.compute_weight(item, residual=1.0))
        w_hyp.append(c_hyp.compute_weight(item, residual=1.0))

    # Weight should strictly decrease as horizon (age) increases
    assert w_exp == sorted(w_exp, reverse=True)
    assert w_hyp == sorted(w_hyp, reverse=True)
    assert w_exp[0] == pytest.approx(1.0, abs=1e-4)


def test_drift_consistency_downweighting():
    horizons = [1, 4, 8]
    c = HypothesisH2_AgeWeightedCorrector(horizons, use_drift_consistency=True)

    base = np.zeros(10)

    # Establish recent short-horizon positive trend
    item_h1 = ForecastFeedbackItem(origin_t=0, horizon=1, target_t=1, y_true=2.0, base_samples=base)
    c.update([item_h1])
    assert c.recent_short_horizon_trend > 0

    # Old feedback from horizon 8 arrives with opposite sign (negative residual)
    item_old = ForecastFeedbackItem(origin_t=0, horizon=8, target_t=8, y_true=-2.0, base_samples=base)
    weight_conflicting = c.compute_weight(item_old, residual=-2.0)

    # Feedback with aligned sign
    item_aligned = ForecastFeedbackItem(origin_t=0, horizon=8, target_t=8, y_true=2.0, base_samples=base)
    weight_aligned = c.compute_weight(item_aligned, residual=2.0)

    # Conflicting feedback should be downweighted by at least 3x
    assert weight_conflicting < weight_aligned / 3.0
