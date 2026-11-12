"""Unit tests for Phase IV frontier theorems and modules.

Covers:
- Theorem 17: Koopman Operator Spectral Cross-Horizon Linearization (Koopman-H3)
- Theorem 18: Differentially Private Cross-Horizon Adaptation (DP-ACHA)
- Theorem 19: Causal Invariant Risk Minimization (CausalIRM-H3)
- Theorem 20: Optimal Transport Wasserstein Barycenter (Wasserstein-H3)
"""

import math
import numpy as np
import pytest

from cita2027.adaptation.koopman_adaptation import (
    OnlineExtendedDynamicModeDecomposition,
    KoopmanCrossHorizonAdapter,
    default_observable_lifting,
)
from cita2027.adaptation.differentially_private_adaptation import (
    DifferentiallyPrivateCrossHorizonAdapter,
)
from cita2027.adaptation.causal_irm_adaptation import (
    CausalEnvironmentDetector,
    CausalIRMCrossHorizonAdapter,
)
from cita2027.adaptation.wasserstein_barycenter_adaptation import (
    WassersteinBarycenterCrossHorizonAdapter,
)


class TestKoopmanSpectralAdaptation:
    """Tests for Theorem 17 Koopman operator spectral dynamics."""

    def test_observable_lifting_dimensions(self):
        x = np.array([0.5])
        lifted = default_observable_lifting(x)
        assert len(lifted) == 5
        assert np.isclose(lifted[0], 0.5)
        assert np.isclose(lifted[1], 0.25)
        assert np.isclose(lifted[2], math.sin(0.5))
        assert np.isclose(lifted[3], math.cos(0.5))
        assert np.isclose(lifted[4], math.tanh(0.5))

    def test_spectral_radius_bound_enforcement(self):
        edmd = OnlineExtendedDynamicModeDecomposition(
            input_dim=1,
            spectral_radius_max=0.90,
        )
        # Feed high-amplitude oscillating inputs
        for t in range(50):
            res = 2.0 * math.sin(0.5 * t)
            info = edmd.update(np.array([res]))
            assert info["spectral_radius"] <= 0.90 + 1e-4

    def test_multi_step_spectral_prediction_and_adaptation(self):
        adapter = KoopmanCrossHorizonAdapter(
            horizons=[1, 2, 4, 8, 16],
            spectral_radius_max=0.90,
        )
        feedback = [
            {"horizon": 1, "y_true": 1.2, "y_pred_loc": 0.2, "y_pred_disp": 1.0},
            {"horizon": 2, "y_true": 1.5, "y_pred_loc": 0.3, "y_pred_disp": 1.0},
        ]
        info = adapter.update_feedback(feedback)
        assert info["num_updates"] == 2

        base_samples = np.random.normal(0.0, 1.0, 100)
        adapted, loc, disp = adapter.adapt_forecast(
            horizon=4,
            base_samples=base_samples,
            base_location=0.0,
            base_dispersion=1.0,
        )
        assert len(adapted) == 100
        assert np.isfinite(loc)
        assert disp > 0.0


class TestDifferentiallyPrivateAdaptation:
    """Tests for Theorem 18 DP-ACHA privacy guarantees and regret bounds."""

    def test_gradient_clipping_bound(self):
        dp = DifferentiallyPrivateCrossHorizonAdapter(
            horizons=[1, 2, 4],
            clipping_norm=1.5,
            epsilon=2.0,
        )
        item = {"horizon": 1, "y_true": 100.0, "y_pred_loc": 0.0, "y_pred_disp": 1.0}
        grad = dp.compute_clipped_gradient(item)
        assert np.linalg.norm(grad) <= 1.5 + 1e-6

    def test_causal_dp_update_and_privacy_accumulation(self):
        dp = DifferentiallyPrivateCrossHorizonAdapter(
            horizons=[1, 2, 4],
            clipping_norm=1.0,
            epsilon=1.0,
            delta=1e-5,
            random_seed=123,
        )
        feedback = [
            {"horizon": 1, "y_true": 0.8, "y_pred_loc": 0.2, "y_pred_disp": 1.0},
            {"horizon": 2, "y_true": 1.0, "y_pred_loc": 0.1, "y_pred_disp": 1.0},
        ]
        info = dp.update_feedback(feedback)
        assert info["num_updates"] == 2
        assert info["noise_norm"] >= 0.0
        assert info["privacy_spent"] > 0.0
        assert info["z_norm"] <= 8.0

        base_samples = np.random.normal(0.5, 1.0, 100)
        adapted, loc, disp = dp.adapt_forecast(1, base_samples, 0.5, 1.0)
        assert len(adapted) == 100
        assert disp > 0.0


class TestCausalIRMAdaptation:
    """Tests for Theorem 19 Causal IRM multi-environment invariance."""

    def test_environment_detector(self):
        detector = CausalEnvironmentDetector(window_size=10)
        # Calm sequence
        for _ in range(8):
            env = detector.observe_and_classify(0.05)
        assert env == 0

        # Turbulent sequence
        for _ in range(8):
            env = detector.observe_and_classify(3.0 * np.random.randn())
        assert env in (1, 2)

    def test_irm_gradient_and_invariant_adaptation(self):
        adapter = CausalIRMCrossHorizonAdapter(
            horizons=[1, 2, 4],
            irm_penalty_lambda=2.0,
            learning_rate=0.05,
        )
        feedback = [
            {"horizon": 1, "y_true": 0.5, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
            {"horizon": 2, "y_true": -0.8, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
            {"horizon": 4, "y_true": 2.5, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
        ]
        info = adapter.update_feedback(feedback)
        assert info["num_updates"] == 3
        assert info["erm_loss"] >= 0.0
        assert info["irm_penalty"] >= 0.0
        assert info["z_norm"] <= 8.0

        base_samples = np.random.normal(0.0, 1.0, 100)
        adapted, loc, disp = adapter.adapt_forecast(4, base_samples, 0.0, 1.0)
        assert len(adapted) == 100
        assert disp > 0.0


class TestWassersteinBarycenterAdaptation:
    """Tests for Theorem 20 Wasserstein Barycenter quantile monotonicity and morphing."""

    def test_quantile_monotonicity(self):
        adapter = WassersteinBarycenterCrossHorizonAdapter(
            horizons=[1, 2, 4],
            num_quantiles=30,
        )
        quantiles = adapter.compute_wasserstein_barycenter_quantiles()
        # Verify strict non-decreasing order
        diffs = np.diff(quantiles)
        assert np.all(diffs >= -1e-8)

    def test_causal_feedback_and_sample_morphing(self):
        adapter = WassersteinBarycenterCrossHorizonAdapter(
            horizons=[1, 2, 4],
            num_quantiles=40,
        )
        feedback = [
            {"horizon": 1, "y_true": 1.2, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
            {"horizon": 2, "y_true": 1.5, "y_pred_loc": 0.1, "y_pred_disp": 1.0},
        ]
        info = adapter.update_feedback(feedback)
        assert info["num_updates"] == 2
        assert info["barycenter_dispersion"] > 0.0

        base_samples = np.random.normal(0.0, 1.0, 100)
        adapted, loc, disp = adapter.adapt_forecast(2, base_samples, 0.0, 1.0)
        assert len(adapted) == 100
        assert disp > 0.0
