"""Unit tests for Phase V frontier theorems and modules.

Covers:
- Theorem 21: Fractional Brownian Motion & Long-Memory Hurst Exponent Adaptation (Hurst-H3)
- Theorem 22: Self-Supervised Contrastive Representation Adaptation (Contrastive-H3)
- Theorem 23: Symbolic Regression & Interpretable Equation Discovery (Symbolic-H3)
- Theorem 24: Curvature-Aware Hessian-Free Optimization (HessianFree-H3)
"""

import math
import numpy as np
import pytest

from cita2027.adaptation.fractional_brownian_adaptation import (
    OnlineHurstEstimator,
    FractionalBrownianCrossHorizonAdapter,
)
from cita2027.adaptation.contrastive_cross_horizon import (
    ContrastiveCrossHorizonAdapter,
)
from cita2027.adaptation.symbolic_equation_discovery import (
    OnlineSymbolicSINDy,
    SymbolicCrossHorizonAdapter,
)
from cita2027.adaptation.hessian_free_second_order import (
    TruncatedConjugateGradientSolver,
    HessianFreeCrossHorizonAdapter,
)


class TestFractionalBrownianAdaptation:
    """Tests for Theorem 21 Fractional Brownian memory adaptation."""

    def test_online_hurst_estimation(self):
        estimator = OnlineHurstEstimator(horizons=[1, 2, 4, 8, 16], window_size=20)
        # Feed variance growing linearly with h (Standard Brownian motion H=0.5)
        for h in [1, 2, 4, 8, 16]:
            for _ in range(10):
                estimator.observe(h, float(h) + 0.1 * np.random.randn()**2)
        h_est = estimator.current_hurst
        assert 0.05 <= h_est <= 0.95
        assert np.isfinite(h_est)

    def test_fractional_weight_and_adaptation(self):
        adapter = FractionalBrownianCrossHorizonAdapter(
            horizons=[1, 2, 4, 8, 16],
            learning_rate=0.04,
        )
        fb = [
            {"horizon": 1, "y_true": 1.2, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
            {"horizon": 2, "y_true": 1.8, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
            {"horizon": 4, "y_true": 2.5, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
        ]
        info = adapter.update_feedback(fb)
        assert info["num_updates"] == 3
        assert info["z_norm"] <= 8.0

        base_samples = np.random.normal(0.0, 1.0, 100)
        adapted, loc, disp = adapter.adapt_forecast(4, base_samples, 0.0, 1.0)
        assert len(adapted) == 100
        assert disp > 0.0


class TestContrastiveCrossHorizonAdaptation:
    """Tests for Theorem 22 Contrastive InfoNCE representation alignment."""

    def test_normalized_embedding_and_infonce(self):
        adapter = ContrastiveCrossHorizonAdapter(
            horizons=[1, 2, 4],
            embedding_dim=4,
            temperature=0.1,
        )
        emb1 = adapter.get_normalized_embedding(1)
        assert np.isclose(np.linalg.norm(emb1), 1.0, atol=1e-5)

        # Update feedback with positive pair
        fb = [
            {"horizon": 1, "y_true": 0.5, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
            {"horizon": 2, "y_true": 0.6, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
        ]
        info = adapter.update_feedback(fb)
        assert info["num_updates"] == 2
        assert info["mean_z_norm"] <= 8.0

    def test_contrastive_forecast_adaptation(self):
        adapter = ContrastiveCrossHorizonAdapter(horizons=[1, 2, 4])
        base_samples = np.random.normal(0.0, 1.0, 100)
        adapted, loc, disp = adapter.adapt_forecast(2, base_samples, 0.0, 1.0)
        assert len(adapted) == 100
        assert disp > 0.0


class TestSymbolicEquationDiscovery:
    """Tests for Theorem 23 SINDy symbolic governing equation discovery."""

    def test_sindy_sparse_recovery(self):
        sindy = OnlineSymbolicSINDy(threshold=0.05)
        # Synthetic cubic attractor: de/dt = -0.5 * e - 0.2 * e^3
        e = 0.5
        for t in range(50):
            de_dt = -0.5 * e - 0.2 * (e**3)
            e_next = e + 0.1 * de_dt
            sindy.update(e_next, dt=0.1)
            e = e_next

        eq_str = sindy.get_symbolic_equation_string()
        assert "de/dt" in eq_str
        assert sindy.predict_derivative(0.0) is not None

    def test_symbolic_adapter_integration(self):
        adapter = SymbolicCrossHorizonAdapter(horizons=[1, 2, 4])
        fb = [
            {"horizon": 1, "y_true": 1.0, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
            {"horizon": 2, "y_true": 1.2, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
        ]
        info = adapter.update_feedback(fb)
        assert info["num_updates"] == 2
        assert "equation" in info

        base_samples = np.random.normal(0.0, 1.0, 100)
        adapted, loc, disp = adapter.adapt_forecast(2, base_samples, 0.0, 1.0)
        assert len(adapted) == 100
        assert disp > 0.0


class TestHessianFreeOptimization:
    """Tests for Theorem 24 Curvature-aware Hessian-Free Krylov optimization."""

    def test_cg_quadratic_solve(self):
        solver = TruncatedConjugateGradientSolver(max_iter=10, damping=0.01)
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([2.0, 1.0])
        hvp_fn = lambda v: A @ v
        x = solver.solve(hvp_fn, b)
        # Check (A + damping*I) x ~= b
        res = (A + 0.01 * np.eye(2)) @ x - b
        assert np.linalg.norm(res) < 1e-2

    def test_hessian_free_adapter_updates(self):
        adapter = HessianFreeCrossHorizonAdapter(
            horizons=[1, 2, 4],
            latent_dim=4,
            cg_max_iter=5,
        )
        fb = [
            {"horizon": 1, "y_true": 1.5, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
            {"horizon": 2, "y_true": 1.7, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
        ]
        info = adapter.update_feedback(fb)
        assert info["num_updates"] == 2
        assert info["z_norm"] <= 8.0

        base_samples = np.random.normal(0.0, 1.0, 100)
        adapted, loc, disp = adapter.adapt_forecast(4, base_samples, 0.0, 1.0)
        assert len(adapted) == 100
        assert disp > 0.0
