"""
Unit Tests for Active Information-Theoretic Feedback Selection.
"""

import numpy as np
import pytest

from cita2027.adaptation.active_feedback_selection import ActiveFeedbackSelector
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


def test_active_feedback_cost_computation():
    selector = ActiveFeedbackSelector(cost_policy="linear_horizon")
    cost_h1 = selector.compute_cost(1)
    cost_h16 = selector.compute_cost(16)
    assert cost_h1 < cost_h16, "Long horizon should cost more in linear model"


def test_active_feedback_budget_constraint_enforcement():
    selector = ActiveFeedbackSelector(cost_policy="uniform")
    items = [
        ForecastFeedbackItem(
            origin_t=0,
            horizon=h,
            target_t=10,
            y_true=10.0 + h,
            base_samples=np.ones(10) * 10.0,
        )
        for h in [1, 2, 4, 8, 16]
    ]

    # With uniform cost=1.0 and budget=2.0, at most 2 items should be selected
    selected, metrics = selector.select_feedback(
        items,
        current_t=10,
        budget=2.0,
        strategy="active_information_gain",
    )

    assert len(selected) <= 2
    assert metrics["total_cost"] <= 2.0 + 1e-6
    assert metrics["retained_ratio"] <= 0.41


def test_active_feedback_full_unconstrained():
    selector = ActiveFeedbackSelector()
    items = [
        ForecastFeedbackItem(
            origin_t=0,
            horizon=h,
            target_t=5,
            y_true=1.0,
            base_samples=np.ones(5),
        )
        for h in [1, 2, 4]
    ]

    selected, metrics = selector.select_feedback(
        items, current_t=5, budget=100.0, strategy="full_unconstrained"
    )
    assert len(selected) == len(items)
    assert metrics["retained_ratio"] == 1.0
