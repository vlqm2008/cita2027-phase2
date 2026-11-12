"""
Unit Tests for Jump-Diffusion Cross-Horizon Adaptation (JumpDiff-H3).
"""

import numpy as np
import pytest

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.jump_diffusion_adaptation import (
    JumpDiffusionCrossHorizonAdapter,
    JumpDiffusionOnlineFilter,
)


def test_jump_diffusion_filter_detection():
    filt = JumpDiffusionOnlineFilter(window_size=10, jump_threshold=0.3)

    # Feed smooth continuous residuals
    for _ in range(15):
        vol, jump_val, is_jump = filt.observe_residual(0.2)
        assert not is_jump

    # Introduce sporadic isolated jump shocks (characteristic of Poisson jumps)
    for _ in range(5):
        filt.observe_residual(15.0)  # Sudden jump spike
        filt.observe_residual(0.1)   # Normal diffusion
        filt.observe_residual(0.1)

    # Jump intensity should now be significantly elevated
    assert filt.jump_intensity_est > 0.1


def test_jump_diffusion_adapter_damping_and_cushion():
    horizons = [1, 2, 4, 8, 16]
    adapter = JumpDiffusionCrossHorizonAdapter(
        horizons=horizons,
        latent_dim=2,
        jump_damping_kappa=3.0,
        jump_cushion_beta=2.0,
    )

    # 1. Smooth feedback
    items_smooth = [
        ForecastFeedbackItem(
            origin_t=0,
            horizon=h,
            target_t=h,
            y_true=1.0,
            base_samples=np.full(10, 0.95),
        )
        for h in horizons
    ]
    res_smooth = adapter.update_with_feedback(items_smooth, current_t=10)
    assert res_smooth["mean_jump_intensity"] < 0.2
    assert res_smooth["damping_applied"] > 0.6

    loc1, scale1 = adapter.get_corrections(16)
    assert scale1 > 0.0

    # 2. Large jump shock feedback
    items_jump = [
        ForecastFeedbackItem(
            origin_t=5,
            horizon=h,
            target_t=15,
            y_true=50.0,
            base_samples=np.full(10, 0.0),
        )
        for h in horizons
    ]
    res_jump = adapter.update_with_feedback(items_jump, current_t=20)

    # Verify state norm is bounded
    assert res_jump["state_norm"] <= 8.0

    # Verify scale cushion
    loc2, scale2 = adapter.get_corrections(16)
    assert scale2 >= scale1
    assert 0.1 <= scale2 <= 10.0


def test_jump_diffusion_adapted_samples():
    horizons = [1, 4]
    adapter = JumpDiffusionCrossHorizonAdapter(horizons=horizons)

    base_samples = np.array([10.0, 11.0, 12.0, 13.0])
    adapted = adapter.get_adapted_samples(horizon=4, base_samples=base_samples)

    assert len(adapted) == len(base_samples)
    assert not np.isnan(adapted).any()
    assert not np.isinf(adapted).any()
