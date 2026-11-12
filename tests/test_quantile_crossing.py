"""
Unit Tests for Online Quantile Crossing Rectification (Quantile-H3).
"""

import numpy as np
import pytest

from cita2027.adaptation.quantile_crossing_rectifier import (
    OnlineQuantileCrossingRectifier,
    pool_adjacent_violators,
)


def test_pool_adjacent_violators_monotonicity():
    # Sequence with crossings: [3.0, 1.0, 2.0, 5.0, 4.0]
    raw = [3.0, 1.0, 2.0, 5.0, 4.0]
    pava = pool_adjacent_violators(raw)

    assert len(pava) == len(raw)
    for i in range(len(pava) - 1):
        assert pava[i] <= pava[i + 1] + 1e-12, "PAVA must produce non-decreasing sequence"


def test_quantile_rectifier_level_and_horizon_monotonicity():
    horizons = [1, 4, 16]
    alphas = [0.10, 0.50, 0.90]
    rectifier = OnlineQuantileCrossingRectifier(
        horizons=horizons,
        alpha_levels=alphas,
        enforce_horizon_monotonicity=True,
    )

    # Intentionally inverted quantiles
    raw_dict = {
        1: {0.10: 5.0, 0.50: 3.0, 0.90: 2.0},  # Crossing!
        4: {0.10: 2.0, 0.50: 4.0, 0.90: 8.0},
        16: {0.10: 1.0, 0.50: 4.0, 0.90: 6.0},  # Narrower than h=4!
    }

    rectified = rectifier.rectify_quantiles(raw_dict)

    # 1. Level-wise monotonicity
    for h in horizons:
        q_10 = rectified[h][0.10]
        q_50 = rectified[h][0.50]
        q_90 = rectified[h][0.90]
        assert q_10 <= q_50 <= q_90, f"Quantiles at h={h} must be strictly non-crossing"

    # 2. Horizon-wise interval width monotonicity
    width_1 = rectified[1][0.90] - rectified[1][0.10]
    width_4 = rectified[4][0.90] - rectified[4][0.10]
    width_16 = rectified[16][0.90] - rectified[16][0.10]

    assert width_1 <= width_4 + 1e-6
    assert width_4 <= width_16 + 1e-6, "Horizon intervals must be non-decreasing"

    assert rectifier.quantile_crossing_violation_rate == 1.0
