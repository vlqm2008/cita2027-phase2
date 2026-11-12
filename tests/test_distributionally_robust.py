"""Unit tests for Wasserstein Distributionally Robust Adaptation (W-DRO-H3)."""
from __future__ import annotations

import math
import numpy as np
import pytest

from cita2027.adaptation.distributionally_robust_adaptation import (
    WassersteinDRO_CrossHorizonAdapter,
)
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


def test_wdro_initialization():
    horizons = [1, 4, 8, 16]
    adapter = WassersteinDRO_CrossHorizonAdapter(horizons=horizons, wasserstein_radius_eps=0.05)
    assert adapter.wasserstein_radius_eps == 0.05
    assert len(adapter.rolling_dispersion) == 4


def test_wdro_clips_extreme_outlier():
    horizons = [1, 4]
    adapter = WassersteinDRO_CrossHorizonAdapter(horizons=horizons, huber_tuning_k=2.0)
    adapter.rolling_dispersion[1] = 1.0  # Threshold = 2.0

    raw_err = 50.0  # Extreme outlier
    clipped_err = adapter.robust_influence_function(raw_err, horizon=1)
    assert clipped_err == 2.0
    assert adapter.outlier_rejection_count[1] == 1

    # Normal error should not be clipped
    normal_err = 1.2
    assert adapter.robust_influence_function(normal_err, horizon=1) == 1.2


def test_wdro_prevents_latent_state_explosion_vs_standard_h3():
    horizons = [1, 4, 8]
    adapter_robust = WassersteinDRO_CrossHorizonAdapter(horizons=horizons, huber_tuning_k=2.0)
    adapter_standard = HypothesisH3_OverlapAwareCorrector(horizons=horizons)

    # Contaminated feedback batch containing massive 100-sigma spike at horizon 1
    corrupted_batch = [
        ForecastFeedbackItem(
            origin_t=0,
            horizon=1,
            target_t=1,
            y_true=100.0,  # Corrupted sensor glitch
            base_samples=np.array([0.0, 0.1, -0.1]),
        )
    ]

    adapter_robust.update(corrupted_batch)
    adapter_standard.update(corrupted_batch)

    norm_robust = float(np.linalg.norm(adapter_robust.shared_state))
    norm_standard = float(np.linalg.norm(adapter_standard.shared_state))

    # W-DRO should significantly limit state shock
    assert norm_robust < norm_standard
    assert norm_robust <= 1.25
