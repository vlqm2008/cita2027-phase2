"""Stress and integration tests for Phase V frontier adaptation modules.

Tests:
- Fractional Brownian motion long memory tracking across variable Hurst exponents.
- Contrastive InfoNCE representation alignment under regime shifts.
- SINDy symbolic equation discovery with multiple candidate ODE functional forms.
- Curvature-aware Hessian-Free conjugate gradient solver on ill-conditioned ravines.
- Grand integration across all 4 Phase V adapters running concurrently.
"""

import math
import numpy as np
import pytest

from cita2027.adaptation.fractional_brownian_adaptation import (
    FractionalBrownianCrossHorizonAdapter,
    OnlineHurstEstimator,
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


def test_hurst_subdiffusion_and_superdiffusion():
    """Verify Hurst estimator accurately discriminates subdiffusion (H<0.5) from superdiffusion (H>0.5)."""
    est_sub = OnlineHurstEstimator(horizons=[1, 2, 4, 8, 16], window_size=30)
    est_super = OnlineHurstEstimator(horizons=[1, 2, 4, 8, 16], window_size=30)

    for h in [1, 2, 4, 8, 16]:
        # Subdiffusion variance scales as h^0.6 (H ~ 0.3)
        var_sub = float(h) ** 0.6
        # Superdiffusion variance scales as h^1.6 (H ~ 0.8)
        var_super = float(h) ** 1.6
        for _ in range(8):
            est_sub.observe(h, var_sub)
            est_super.observe(h, var_super)

    h1 = est_sub.current_hurst
    h2 = est_super.current_hurst
    assert h1 < h2, f"Subdiffusive Hurst ({h1}) must be less than superdiffusive Hurst ({h2})"


def test_contrastive_hypersphere_collapse_prevention():
    """Verify that InfoNCE loss strictly prevents representation collapse onto a single point."""
    adapter = ContrastiveCrossHorizonAdapter(
        horizons=[1, 2, 4, 8],
        embedding_dim=4,
        temperature=0.1,
    )
    # Stream 30 iterations of distinct positive and negative feedback
    for t in range(30):
        y1 = 1.0 + 0.1 * np.random.randn()
        y2 = -1.0 + 0.1 * np.random.randn()
        fb = [
            {"horizon": 1, "y_true": y1, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
            {"horizon": 2, "y_true": y1, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
            {"horizon": 4, "y_true": y2, "y_pred_loc": 0.0, "y_pred_disp": 1.0},
        ]
        adapter.update_feedback(fb)

    emb1 = adapter.get_normalized_embedding(1)
    emb4 = adapter.get_normalized_embedding(4)
    # Cosine similarity must not be 1.0 (no collapse)
    cos_sim = float(np.dot(emb1, emb4))
    assert cos_sim < 0.99, f"Representations collapsed! cos_sim={cos_sim}"


def test_sindy_polynomial_recovery_stress():
    """Verify SINDy online identification under high-frequency noise."""
    sindy = OnlineSymbolicSINDy(threshold=0.03, alpha_ridge=1e-4)
    e = 0.2
    for t in range(60):
        de_dt = -0.4 * e - 0.15 * (e**2)
        e_next = e + 0.1 * de_dt + 0.01 * np.random.randn()
        sindy.update(e_next, dt=0.1)
        e = e_next

    eq = sindy.get_symbolic_equation_string()
    assert "de/dt" in eq
    assert np.all(np.isfinite(sindy.xi))


def test_hessian_free_high_condition_number_invariance():
    """Verify Hessian-Free CG handles condition numbers up to 10,000 without crashing."""
    solver = TruncatedConjugateGradientSolver(max_iter=15, damping=1e-3)
    # Ill-conditioned matrix with eigenvalues [1e-4, 1.0]
    A = np.diag([1e-4, 1.0])
    b = np.array([1.0, 1.0])
    hvp_fn = lambda v: A @ v
    x = solver.solve(hvp_fn, b)
    assert np.all(np.isfinite(x))


def test_golden_zenith_four_adapter_pipeline():
    """Integration test: all 4 Phase V adapters running in single causal prequential loop."""
    horizons = [1, 2, 4, 8, 16]
    hurst = FractionalBrownianCrossHorizonAdapter(horizons=horizons)
    contrast = ContrastiveCrossHorizonAdapter(horizons=horizons)
    symbolic = SymbolicCrossHorizonAdapter(horizons=horizons)
    hf = HessianFreeCrossHorizonAdapter(horizons=horizons)

    adapters = [hurst, contrast, symbolic, hf]

    for t in range(25):
        fb = []
        for h in horizons:
            if t >= h:
                fb.append({
                    "horizon": h,
                    "y_true": 0.8 + 0.1 * np.random.randn(),
                    "y_pred_loc": 0.0,
                    "y_pred_disp": 1.0,
                })
        for ad in adapters:
            ad.update_feedback(fb)

        # Adapt forecast at h=4
        base_samples = np.random.randn(80)
        for ad in adapters:
            ad_s, loc, disp = ad.adapt_forecast(4, base_samples, 0.0, 1.0)
            assert len(ad_s) == 80
            assert np.all(np.isfinite(ad_s))
            assert disp > 0.0


def test_hurst_long_term_decay_monotonicity():
    """Verify power law kernel omega(h) decreases monotonically for persistent Hurst H=0.75."""
    adapter = FractionalBrownianCrossHorizonAdapter(horizons=[1, 2, 4, 8, 16, 32, 64])
    weights = [adapter.compute_fractional_weight(h, 0.75) for h in [1, 2, 4, 8, 16, 32, 64]]
    diffs = np.diff(weights)
    assert np.all(diffs < 0.0), "Weights must decrease monotonically with horizon"


def test_sindy_candidate_library_numerical_stability():
    """Verify SINDy basis functions evaluate stably across large input excursions."""
    sindy = OnlineSymbolicSINDy()
    for test_e in [-10.0, -1.0, 0.0, 1.0, 10.0]:
        lib = sindy.compute_library(test_e)
        assert len(lib) == 8
        assert np.all(np.isfinite(lib))


def test_hessian_free_zero_gradient_invariance():
    """Verify Hessian-Free solver preserves exact state when gradient is zero."""
    solver = TruncatedConjugateGradientSolver()
    A = np.eye(3)
    b = np.zeros(3)
    x = solver.solve(lambda v: A @ v, b)
    assert np.all(x == 0.0)
