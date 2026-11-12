"""Unit tests for Spatio-Temporal Cross-Horizon Adaptation (ST-H3)."""
from __future__ import annotations

import math
import numpy as np
import pytest

from cita2027.adaptation.spatiotemporal_adaptation import (
    SpatialFeedbackItem,
    SpatioTemporalCrossHorizonAdapter,
)


def test_st_adapter_initialization():
    horizons = [1, 4, 8]
    adapter = SpatioTemporalCrossHorizonAdapter(num_nodes=4, horizons=horizons, latent_dim=2)
    assert adapter.num_nodes == 4
    assert adapter.Z.shape == (4, 2)
    assert adapter.L.shape == (4, 4)
    # Normalized Laplacian diagonal should be close to 1
    assert np.allclose(np.diag(adapter.L), 1.0)


def test_st_adapter_adapt():
    adapter = SpatioTemporalCrossHorizonAdapter(num_nodes=3, horizons=[1, 4], latent_dim=2)
    samples = np.array([10.0, 11.0, 12.0])
    adapted = adapter.adapt(node_id=1, origin_t=0, horizon=4, base_samples=samples)
    assert adapted.shape == samples.shape
    assert not np.isnan(adapted).any()


def test_spatial_diffusion_propagates_to_neighbors_without_direct_feedback():
    adapter = SpatioTemporalCrossHorizonAdapter(
        num_nodes=3, horizons=[1, 4], latent_dim=2, lambda_graph=0.3, lr_spatial=0.1
    )

    # Shock observed ONLY at node 0
    feedback_batch = [
        SpatialFeedbackItem(
            node_id=0,
            origin_t=0,
            horizon=1,
            target_t=1,
            y_true=5.0,  # Positive shock
            base_samples=np.array([0.0, 0.1, -0.1]),
        )
    ]

    z1_initial = adapter.Z[1].copy()
    adapter.update(feedback_batch)
    z0_after = adapter.Z[0].copy()

    # Node 0 state should update strongly
    assert np.linalg.norm(z0_after) > 0.0

    # In subsequent step with no feedback anywhere, Laplacian diffusion propagates to node 1!
    adapter.update([])
    z1_after = adapter.Z[1].copy()

    # Node 1 state should have moved towards node 0's state via graph diffusion!
    assert not np.allclose(z1_initial, z1_after)
    assert np.linalg.norm(z1_after) > 0.0
