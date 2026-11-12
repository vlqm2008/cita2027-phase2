"""
Unit Tests for Nonlinear Kernelized Cross-Horizon Adaptation (Kernel-H3).
"""

import numpy as np
import pytest

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.kernel_adaptation import (
    HypothesisKernelH3_Corrector,
    RandomFourierFeaturesEmbedding,
)


def test_rff_embedding_dimensions_and_determinism():
    rff1 = RandomFourierFeaturesEmbedding(input_dim=3, num_features=16, seed=42)
    rff2 = RandomFourierFeaturesEmbedding(input_dim=3, num_features=16, seed=42)

    x = np.array([0.5, 1.2, -0.3])
    phi1 = rff1.transform(x)
    phi2 = rff2.transform(x)

    assert phi1.shape == (16,)
    assert np.allclose(phi1, phi2), "RFF transform must be deterministic given seed"
    assert np.all(np.isfinite(phi1)), "Features must be finite"


def test_kernel_h3_corrections_and_adaptation():
    horizons = [1, 2, 4, 8, 16]
    corrector = HypothesisKernelH3_Corrector(horizons=horizons, rff_dim=16, seed=42)

    delta_mu, scale_mult = corrector.get_corrections(4)
    assert np.isfinite(delta_mu)
    assert scale_mult > 0.0, "Scale multiplier must be strictly positive"

    base_samples = np.random.normal(loc=10.0, scale=2.0, size=100)
    adapted = corrector.adapt_samples(base_samples, base_location=10.0, h=4)
    assert len(adapted) == 100
    assert np.all(np.isfinite(adapted))


def test_kernel_h3_feedback_update_and_boundedness():
    horizons = [1, 2, 4, 8, 16]
    corrector = HypothesisKernelH3_Corrector(horizons=horizons, rff_dim=16, max_state_norm=5.0)

    # Feed 20 feedback items with persistent shift
    for t in range(20):
        items = [
            ForecastFeedbackItem(
                origin_t=max(0, t - h),
                horizon=h,
                target_t=t,
                y_true=15.0 + np.random.normal(0, 0.5),
                base_samples=np.random.normal(10.0, 1.0, size=50),
            )
            for h in [1, 2]
        ]
        info = corrector.update_with_feedback(items, current_t=t)
        assert info["effective_k"] > 0.0
        assert info["state_norm_mean"] <= 5.0 + 1e-5, "State norm must be bounded by projection"
