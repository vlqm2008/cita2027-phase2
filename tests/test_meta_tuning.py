"""Unit tests for Self-Tuning Online Meta-Controller (Meta-H3)."""
from __future__ import annotations

import math
import numpy as np
import pytest

from cita2027.adaptation.meta_tuning import (
    HyperparameterConfig,
    MetaAdaptiveH3Controller,
)
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


def test_meta_controller_initialization():
    horizons = [1, 4, 8, 16]
    controller = MetaAdaptiveH3Controller(horizons=horizons)
    assert controller.M == 5
    assert len(controller.weights) == 5
    assert math.isclose(np.sum(controller.weights), 1.0)
    assert len(controller.experts) == 5


def test_meta_controller_adapt():
    horizons = [1, 4]
    controller = MetaAdaptiveH3Controller(horizons=horizons)
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    adapted_stochastic = controller.adapt(origin_t=0, horizon=1, base_samples=samples)
    adapted_mixture = controller.adapt_deterministic_mixture(origin_t=0, horizon=1, base_samples=samples)

    assert adapted_stochastic.shape == samples.shape
    assert adapted_mixture.shape == samples.shape
    assert not np.isnan(adapted_stochastic).any()
    assert not np.isnan(adapted_mixture).any()


def test_meta_weights_update_favors_better_expert():
    horizons = [1, 4]
    cfg1 = HyperparameterConfig("Fast", gamma_age=0.40, overlap_discount_factor=0.3, lr_shared=0.15, lr_local=0.05)
    cfg2 = HyperparameterConfig("Slow", gamma_age=0.01, overlap_discount_factor=0.9, lr_shared=0.01, lr_local=0.01)
    controller = MetaAdaptiveH3Controller(horizons=horizons, configs=[cfg1, cfg2], eta_meta=2.0, entropy_floor=0.01)

    # Simulate abrupt shift where Fast expert adapts faster
    feedback_batch = [
        ForecastFeedbackItem(
            origin_t=0,
            horizon=1,
            target_t=1,
            y_true=10.0,  # Huge shift from 0
            base_samples=np.array([0.0, 0.1, -0.1]),
            adapted_samples=np.array([0.0, 0.1, -0.1]),
        )
    ]

    w_before = controller.weights.copy()
    controller.update(feedback_batch)
    w_after = controller.weights.copy()

    # Probability weights must sum to 1.0
    assert math.isclose(np.sum(w_after), 1.0)
    # The weights should have shifted
    assert not np.allclose(w_before, w_after)
    # History should be logged
    assert len(controller.history_weights) == 1
    assert len(controller.history_entropy) == 1
