"""
Unit Tests for Federated Cross-Horizon Adaptation (Fed-ACHA).
"""

import numpy as np
import pytest

from cita2027.adaptation.federated_adaptation import (
    FederatedAgentNode,
    FederatedCrossHorizonCoordinator,
)
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


def test_federated_node_local_adaptation():
    horizons = [1, 2, 4, 8]
    node = FederatedAgentNode(node_id=0, horizons=horizons, latent_dim=2, local_lr=0.05)

    # Initial state should be zero
    assert np.allclose(node.accumulated_delta_z, 0.0)
    assert node.unreported_updates == 0

    items = [
        ForecastFeedbackItem(
            origin_t=0,
            horizon=h,
            target_t=h,
            y_true=5.0 + 0.1 * h,
            base_samples=np.full(20, 4.0),
        )
        for h in horizons
    ]

    metrics = node.step_local_adaptation(items, current_t=8)
    assert metrics["k_eff"] > 0.0
    assert node.unreported_updates == len(horizons)
    assert np.linalg.norm(node.accumulated_delta_z) > 0.0

    delta_z, k_sum, n_up = node.emit_federated_payload()
    assert n_up == len(horizons)
    assert k_sum > 0.0
    assert len(delta_z) == 2
    # After emit, buffer is reset
    assert node.unreported_updates == 0
    assert np.allclose(node.accumulated_delta_z, 0.0)


def test_federated_coordinator_aggregation():
    horizons = [1, 2, 4, 8]
    coordinator = FederatedCrossHorizonCoordinator(
        num_nodes=3,
        horizons=horizons,
        latent_dim=2,
        global_learning_rate=1.0,
    )

    # Each node receives local feedback
    for i, node in enumerate(coordinator.nodes):
        items = [
            ForecastFeedbackItem(
                origin_t=0,
                horizon=h,
                target_t=h,
                y_true=2.0 * (i + 1),
                base_samples=np.full(15, 1.0),
            )
            for h in horizons
        ]
        node.step_local_adaptation(items, current_t=8)

    # Aggregate round across all 3 nodes
    res = coordinator.aggregate_federated_round()
    assert res["num_participants"] == 3
    assert res["total_k_eff"] > 0.0
    assert res["global_z_norm"] > 0.0

    # Ensure nodes are synchronized
    for node in coordinator.nodes:
        assert np.allclose(node.local_adapter.shared_state, coordinator.global_z)


def test_federated_stability_and_projection():
    horizons = [1, 2, 4]
    coordinator = FederatedCrossHorizonCoordinator(
        num_nodes=2,
        horizons=horizons,
        latent_dim=2,
        global_learning_rate=100.0,  # Extreme learning rate to test projection
    )

    for node in coordinator.nodes:
        items = [
            ForecastFeedbackItem(
                origin_t=0,
                horizon=1,
                target_t=1,
                y_true=1000.0,
                base_samples=np.zeros(10),
            )
        ]
        node.step_local_adaptation(items, current_t=1)

    res = coordinator.aggregate_federated_round()
    # Norm must be bounded by 8.0
    assert res["global_z_norm"] <= 8.0 + 1e-6
    for node in coordinator.nodes:
        assert np.linalg.norm(node.local_adapter.shared_state) <= 8.0 + 1e-6
