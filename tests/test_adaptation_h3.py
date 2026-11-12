"""Unit tests for Hypothesis H3 overlap-aware adaptation."""
from __future__ import annotations

import numpy as np
import pytest

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector


def test_effective_overlap_factor_scaling():
    horizons = [1, 4, 8, 16]
    c = HypothesisH3_OverlapAwareCorrector(horizons, overlap_discount_factor=0.7)

    base = np.zeros(10)

    # Single item batch: K_eff must be 1.0
    item1 = ForecastFeedbackItem(origin_t=0, horizon=1, target_t=1, y_true=1.0, base_samples=base)
    assert c.compute_effective_overlap_factor([item1]) == 1.0

    # Multi-item batch with overlapping horizons: K_eff must scale > K
    item2 = ForecastFeedbackItem(origin_t=0, horizon=4, target_t=4, y_true=1.2, base_samples=base)
    k_eff_2 = c.compute_effective_overlap_factor([item1, item2])
    assert k_eff_2 > 2.0  # Due to positive overlap cross-term

    item3 = ForecastFeedbackItem(origin_t=0, horizon=8, target_t=8, y_true=1.5, base_samples=base)
    k_eff_3 = c.compute_effective_overlap_factor([item1, item2, item3])
    assert k_eff_3 > k_eff_2


def test_step_norm_boundedness():
    horizons = [1, 4, 8, 16]
    c = HypothesisH3_OverlapAwareCorrector(horizons)

    base = np.zeros(10)
    items = [
        ForecastFeedbackItem(origin_t=0, horizon=h, target_t=h, y_true=5.0, base_samples=base)
        for h in horizons
    ]

    c.update(items)
    assert len(c.state_step_history) == 1
    # Bounded update step: should not explode
    assert c.state_step_history[0] < 2.0
    assert np.isfinite(c.shared_state).all()
