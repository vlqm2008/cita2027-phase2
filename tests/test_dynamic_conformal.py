"""Unit tests for Asynchronous Cross-Horizon Dynamic Conformal Calibrator."""
from __future__ import annotations

import math
import numpy as np
import pytest

from cita2027.adaptation.dynamic_conformal_calibrator import (
    AsynchronousCrossHorizonConformalCalibrator,
    ConformalInterval,
)


def test_conformal_calibrator_initialization():
    horizons = [1, 4, 8, 16]
    calibrator = AsynchronousCrossHorizonConformalCalibrator(horizons=horizons, alpha=0.10)
    assert calibrator.horizons == [1, 4, 8, 16]
    assert math.isclose(calibrator.target_coverage, 0.90)
    for h in horizons:
        assert calibrator.q[h] == 1.645


def test_conformal_interval_generation():
    calibrator = AsynchronousCrossHorizonConformalCalibrator(horizons=[1, 4], alpha=0.10)
    samples = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    interval = calibrator.calibrate_interval(origin_t=5, horizon=4, adapted_samples=samples)
    assert isinstance(interval, ConformalInterval)
    assert interval.horizon == 4
    assert interval.origin_t == 5
    assert interval.target_t == 9
    assert interval.center == 12.0
    assert interval.lower < interval.center < interval.upper
    assert interval.width > 0


def test_conformal_robbins_monro_and_cross_transfer():
    horizons = [1, 4, 8]
    calibrator = AsynchronousCrossHorizonConformalCalibrator(
        horizons=horizons, alpha=0.10, eta_conformal=0.10, gamma_transfer=0.05
    )
    samples = np.array([0.0, 0.1, -0.1])
    interval_h1 = calibrator.calibrate_interval(origin_t=0, horizon=1, adapted_samples=samples)

    q4_before = calibrator.q[4]
    q8_before = calibrator.q[8]

    # Extreme outlier causing coverage breach at horizon 1
    y_true_outlier = 100.0
    rec = calibrator.update_with_feedback(horizon=1, y_true=y_true_outlier, interval=interval_h1)

    assert rec["is_covered"] == 0.0
    # Horizon 1 quantile should increase
    assert calibrator.q[1] > 1.645
    # Horizon 4 and 8 quantiles should also proactively widen due to cross-horizon transfer!
    assert calibrator.q[4] > q4_before
    assert calibrator.q[8] > q8_before


def test_conformal_coverage_tracks_target_coverage():
    np.random.seed(42)
    horizons = [1, 4]
    alpha = 0.15  # Target 85% coverage
    calibrator = AsynchronousCrossHorizonConformalCalibrator(
        horizons=horizons, alpha=alpha, eta_conformal=0.08
    )

    T = 500
    for t in range(T):
        for h in horizons:
            # Simulated true distribution with slow drift
            true_mean = np.sin(t / 50.0) * 2.0
            y_t = np.random.normal(loc=true_mean, scale=1.0)

            # Predictive samples
            samples = np.random.normal(loc=true_mean, scale=1.0, size=50)
            interval = calibrator.calibrate_interval(origin_t=t, horizon=h, adapted_samples=samples)
            calibrator.update_with_feedback(horizon=h, y_true=y_t, interval=interval)

    cov_h1 = calibrator.get_empirical_coverage(1)
    cov_h4 = calibrator.get_empirical_coverage(4)

    # Empirical coverage should be within +/- 5% of target (0.85)
    assert abs(cov_h1 - (1.0 - alpha)) < 0.06
    assert abs(cov_h4 - (1.0 - alpha)) < 0.06
