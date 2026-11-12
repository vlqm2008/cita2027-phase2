"""Unit tests for ultra-long horizon scaling module (H up to 64)."""
from __future__ import annotations

import math
import numpy as np
import pytest

from cita2027.adaptation.ultralong_horizons import (
    MultiScaleDyadicEmbedding,
    HypothesisH3_DyadicUltraLongCorrector,
    run_ultralong_horizon_experiment,
)


def test_dyadic_embedding_properties():
    embedder = MultiScaleDyadicEmbedding(latent_dim=4)
    for h in [1, 2, 4, 8, 16, 32, 64]:
        v = embedder.embed(h)
        assert len(v) == 4
        # Norm should be 1.0 (unit vector)
        assert math.isclose(np.linalg.norm(v), 1.0, rel_tol=1e-5)

    # Higher horizons should have distinct embeddings
    v1 = embedder.embed(1)
    v64 = embedder.embed(64)
    cosine_sim = float(np.dot(v1, v64))
    assert cosine_sim < 0.95  # Distinct representation across 6 octaves


def test_h3_dyadic_corrector_initialization_and_adapt():
    horizons = [1, 2, 4, 8, 16, 32, 64]
    model = HypothesisH3_DyadicUltraLongCorrector(horizons=horizons, latent_dim=4)
    assert len(model.w) == len(horizons)

    samples = np.random.randn(50)
    for h in horizons:
        adapted = model.adapt(0, h, samples)
        assert adapted.shape == samples.shape
        assert not np.isnan(adapted).any()


def test_ultralong_experiment_execution_smoke():
    horizons = [1, 4, 16, 64]
    df = run_ultralong_horizon_experiment(horizons=horizons, length=120, seed=42)
    assert not df.empty
    assert "crps_mean" in df.columns
    assert set(df["horizon"].unique()) == set(horizons)
    assert "H3_Dyadic" in df["model"].unique()
