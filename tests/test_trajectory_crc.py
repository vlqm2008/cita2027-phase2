"""
Unit Tests for Asynchronous Trajectory Conformal Risk Control (AT-CRC).
"""

import numpy as np
import pytest

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.trajectory_conformal_risk_control import AsynchronousTrajectoryCRC


def test_trajectory_crc_prediction_bands():
    horizons = [1, 2, 4, 8]
    crc = AsynchronousTrajectoryCRC(horizons=horizons, alpha_joint=0.10, initial_quantile=2.0)

    base_loc = {h: 5.0 for h in horizons}
    base_disp = {h: 1.0 for h in horizons}

    bands = crc.get_prediction_bands(base_loc, base_disp)
    assert len(bands) == 4
    for h in horizons:
        lower, upper, width = bands[h]
        assert lower < 5.0 < upper
        assert width > 0.0
        # Longer horizons have wider bands due to dispersion scaling
        if h > 1:
            assert bands[h][2] >= bands[1][2]


def test_trajectory_crc_feedback_adaptation():
    horizons = [1, 2, 4]
    crc = AsynchronousTrajectoryCRC(horizons=horizons, alpha_joint=0.10, learning_rate=0.1)

    initial_q = crc.q_joint

    # Simulate 15 steps with extreme breaches to verify quantile expands
    for t in range(15):
        crc.register_forecast(origin_t=t)
        items = [
            ForecastFeedbackItem(
                origin_t=t,
                horizon=h,
                target_t=t + h,
                y_true=25.0,  # Extreme breach relative to base_location 5.0
                base_samples=np.ones(10) * 5.0,
            )
            for h in horizons
        ]
        crc.update_with_feedback(items, current_t=t + 4)

    # Quantile should have adapted upward to cover the breaches
    assert crc.q_joint > initial_q, "Quantile must increase following repeated breaches"
    assert crc.q_joint <= crc.max_quantile
