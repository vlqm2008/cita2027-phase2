"""
Integration Tests for Frontier Adaptation Paradigms (Fed-ACHA, JumpDiff, CausalGraph, Bayes-H3).
"""

import numpy as np
import pytest

from cita2027.adaptation.bayesian_online_posterior import (
    BayesianCrossHorizonPosteriorTracker,
)
from cita2027.adaptation.causal_cross_horizon_discovery import (
    CausalCrossHorizonAdapter,
)
from cita2027.adaptation.distribution_metrics import energy_score
from cita2027.adaptation.federated_adaptation import (
    FederatedCrossHorizonCoordinator,
)
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.jump_diffusion_adaptation import (
    JumpDiffusionCrossHorizonAdapter,
)


def test_end_to_end_causal_jump_synergy():
    horizons = [1, 2, 4, 8]
    jump_adapter = JumpDiffusionCrossHorizonAdapter(horizons=horizons)
    causal_adapter = CausalCrossHorizonAdapter(horizons=horizons)

    # Stream containing both jumps and cross-horizon correlated shifts
    for t in range(1, 15):
        shock = 8.0 if t % 5 == 0 else 0.5
        items = [
            ForecastFeedbackItem(
                origin_t=max(0, t - h),
                horizon=h,
                target_t=t,
                y_true=shock + 0.1 * h,
                base_samples=np.full(10, 0.0),
            )
            for h in horizons
        ]
        res_j = jump_adapter.update_with_feedback(items, current_t=t)
        res_c = causal_adapter.update_with_feedback(items, current_t=t)

        assert res_j["state_norm"] <= 8.0
        assert res_c["state_norm"] <= 8.0

    # Ensure valid predictions generated
    samples = jump_adapter.get_adapted_samples(4, np.ones(10))
    assert len(samples) == 10
    assert not np.isnan(samples).any()


def test_federated_multi_node_consensus_convergence():
    horizons = [1, 2, 4]
    coordinator = FederatedCrossHorizonCoordinator(
        num_nodes=4,
        horizons=horizons,
        latent_dim=2,
        global_learning_rate=0.8,
        consensus_regularization=0.05,
    )

    # 5 rounds of local steps and global aggregation
    for r in range(5):
        for node in coordinator.nodes:
            items = [
                ForecastFeedbackItem(
                    origin_t=0,
                    horizon=h,
                    target_t=r + 1,
                    y_true=2.0 + 0.1 * node.node_id,
                    base_samples=np.full(10, 1.0),
                )
                for h in horizons
            ]
            node.step_local_adaptation(items, current_t=r + 1)
        res = coordinator.aggregate_federated_round()
        assert res["num_participants"] == 4

    # All nodes must have synchronized global states
    for node in coordinator.nodes:
        assert np.allclose(node.local_adapter.shared_state, coordinator.global_z)


def test_bayesian_posterior_energy_score_consistency():
    horizons = [1, 2, 4, 8]
    tracker = BayesianCrossHorizonPosteriorTracker(horizons=horizons, latent_dim=2)

    # Update with some items
    items = [
        ForecastFeedbackItem(
            origin_t=0,
            horizon=h,
            target_t=h,
            y_true=2.0,
            base_samples=np.random.normal(1.5, 0.5, size=25),
        )
        for h in horizons
    ]
    tracker.update_with_feedback(items, current_t=8)

    # Generate multi-horizon trajectory samples
    M = 30
    traj_samples = np.zeros((M, len(horizons)))
    for j, h in enumerate(horizons):
        base_s = np.random.normal(1.5, 0.5, size=M)
        traj_samples[:, j] = tracker.get_adapted_samples(h, base_s)

    y_true_vec = np.array([2.0, 2.0, 2.0, 2.0])
    es = energy_score(y_true=y_true_vec, pred_trajectories=traj_samples)

    assert not np.isnan(es)
    assert not np.isinf(es)
    assert es > 0.0
