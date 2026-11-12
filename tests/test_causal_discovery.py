"""
Unit Tests for Causal Cross-Horizon Directed Graph Discovery (CausalGraph-H3).
"""

import numpy as np
import pytest

from cita2027.adaptation.causal_cross_horizon_discovery import (
    CausalCrossHorizonAdapter,
    OnlineCausalHorizonGraph,
)
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


def test_online_causal_horizon_graph_structure():
    horizons = [1, 2, 4, 8]
    graph = OnlineCausalHorizonGraph(horizons=horizons, l1_reg=0.01, learning_rate=0.05)

    assert graph.adj_matrix.shape == (4, 4)
    assert np.all(np.diagonal(graph.adj_matrix) == 0.0)

    # Simulate causal relationship: h=1 errors drive h=8 errors
    for t in range(20):
        e1 = float(np.sin(t * 0.5))
        e8 = float(0.8 * e1 + np.random.normal(0, 0.05))
        graph.update({1: e1, 2: 0.0, 4: 0.0, 8: e8})

    # Self-loops must remain strictly 0
    assert np.all(np.diagonal(graph.adj_matrix) == 0.0)

    # Weights must sum to 1.0
    weights = graph.get_causal_weights()
    assert len(weights) == 4
    assert np.isclose(sum(weights.values()), 1.0)


def test_causal_cross_horizon_adapter_update():
    horizons = [1, 2, 4]
    adapter = CausalCrossHorizonAdapter(horizons=horizons, latent_dim=2)

    items = [
        ForecastFeedbackItem(
            origin_t=0,
            horizon=h,
            target_t=h,
            y_true=3.0,
            base_samples=np.full(10, 2.0),
        )
        for h in horizons
    ]

    res = adapter.update_with_feedback(items, current_t=5)
    assert res["mean_causal_weight"] > 0.0
    assert res["state_norm"] <= 8.0

    loc, scale = adapter.get_corrections(4)
    assert scale > 0.0
    assert 0.1 <= scale <= 10.0

    samples = adapter.get_adapted_samples(horizon=4, base_samples=np.array([1.0, 2.0, 3.0]))
    assert len(samples) == 3
    assert not np.isnan(samples).any()
