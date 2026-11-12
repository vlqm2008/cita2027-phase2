"""
Multi-Agent Asynchronous Federated Cross-Horizon Adaptation (Fed-ACHA).

Extends single-stream adaptation to decentralized multi-agent sensor networks,
regional electrical microgrids, and multi-exchange financial markets.
Multiple decentralized agents collaboratively estimate a global cross-horizon
representation manifold z_t under heterogeneous communication latencies and
privacy constraints without sharing raw sample trajectories.

Theoretical Foundation (Theorem 6):
    Under heterogeneous communication delays tau_{i, comm} <= D_comm and local
    supervision latencies tau_{i, h} <= H_max, Overlap-Aware Federated Averaging
    achieves an excess regret bound of:
    R_T^{Fed}(z*) <= O( (1/sqrt(N)) * sqrt(T * K_bar_eff) + D_comm * G^2 * sqrt(T) )
    guaranteeing linear speedup in the number of collaborating agents N while
    strictly bounding delay-induced gossip variance.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector


class FederatedAgentNode:
    """
    Decentralized agent node in a federated cross-horizon network.
    """

    def __init__(
        self,
        node_id: int,
        horizons: List[int],
        latent_dim: int = 2,
        local_lr: float = 0.08,
        gamma_age: float = 0.10,
    ):
        self.node_id = node_id
        self.horizons = sorted(list(horizons))
        self.latent_dim = latent_dim
        self.local_lr = local_lr
        self.gamma_age = gamma_age

        self.local_adapter = HypothesisH3_OverlapAwareCorrector(
            horizons=horizons,
            latent_dim=latent_dim,
            lr_shared=local_lr,
            gamma_age=gamma_age,
        )

        self.accumulated_delta_z = np.zeros(latent_dim, dtype=np.float64)
        self.local_k_eff_sum = 0.0
        self.unreported_updates = 0

    def step_local_adaptation(
        self,
        feedback_items: List[ForecastFeedbackItem],
        current_t: int,
    ) -> Dict[str, float]:
        """Process local observations and accumulate local gradient updates."""
        if not feedback_items:
            return {"k_eff": 0.0}

        z_before = np.copy(self.local_adapter.shared_state)
        self.local_adapter.update_with_feedback(feedback_items, current_t=current_t)
        delta = self.local_adapter.shared_state - z_before

        self.accumulated_delta_z += delta
        k_eff = self.local_adapter.compute_effective_overlap_factor(feedback_items)
        self.local_k_eff_sum += float(k_eff)
        self.unreported_updates += len(feedback_items)

        return {"k_eff": float(k_eff), "delta_norm": float(np.linalg.norm(delta))}

    def emit_federated_payload(self) -> Tuple[np.ndarray, float, int]:
        """Export local update delta and effective sample weight, then reset buffer."""
        payload = (np.copy(self.accumulated_delta_z), float(self.local_k_eff_sum), self.unreported_updates)
        self.accumulated_delta_z.fill(0.0)
        self.local_k_eff_sum = 0.0
        self.unreported_updates = 0
        return payload

    def sync_global_state(self, global_z: np.ndarray) -> None:
        """Synchronize local shared manifold with aggregated global state."""
        self.local_adapter.shared_state = np.copy(global_z)


class FederatedCrossHorizonCoordinator:
    """
    Central or peer-to-peer coordinator for Federated Asynchronous Adaptation (Fed-ACHA).
    """

    def __init__(
        self,
        num_nodes: int,
        horizons: List[int],
        latent_dim: int = 2,
        global_learning_rate: float = 1.0,
        consensus_regularization: float = 0.01,
    ):
        self.num_nodes = num_nodes
        self.horizons = sorted(list(horizons))
        self.latent_dim = latent_dim
        self.global_lr = global_learning_rate
        self.consensus_reg = consensus_regularization

        # Global shared representation state
        self.global_z = np.zeros(latent_dim, dtype=np.float64)

        # Instantiate decentralized nodes
        self.nodes = [
            FederatedAgentNode(node_id=i, horizons=horizons, latent_dim=latent_dim)
            for i in range(num_nodes)
        ]

        self.round_history: List[float] = [0.0]

    def aggregate_federated_round(
        self,
        participating_node_ids: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Aggregate local updates from participating nodes using Overlap-Aware FedAvg.
        """
        active_ids = participating_node_ids if participating_node_ids is not None else list(range(self.num_nodes))
        if not active_ids:
            return {"num_participants": 0, "global_z_norm": float(np.linalg.norm(self.global_z))}

        deltas = []
        k_effs = []

        for nid in active_ids:
            delta_z, k_sum, n_up = self.nodes[nid].emit_federated_payload()
            if n_up > 0 and k_sum > 0:
                deltas.append(delta_z)
                k_effs.append(k_sum)

        if not deltas:
            return {"num_participants": 0, "global_z_norm": float(np.linalg.norm(self.global_z))}

        # Overlap-Aware FedAvg weighting: weight_i = K_eff^{(i)} / sum_j K_eff^{(j)}
        total_k_eff = sum(k_effs)
        weighted_delta = np.zeros(self.latent_dim, dtype=np.float64)

        for delta, k_eff in zip(deltas, k_effs):
            weight = k_eff / (total_k_eff + 1e-12)
            weighted_delta += weight * delta

        # Consensus regularization step
        weighted_delta -= self.consensus_reg * self.global_z

        # Apply global step
        self.global_z += self.global_lr * weighted_delta

        # Projection onto compact ball ||z|| <= 8.0
        norm_z = float(np.linalg.norm(self.global_z))
        if norm_z > 8.0:
            self.global_z = (self.global_z / norm_z) * 8.0

        # Broadcast synchronized state back to participating nodes
        for nid in active_ids:
            self.nodes[nid].sync_global_state(self.global_z)

        self.round_history.append(float(np.linalg.norm(self.global_z)))

        return {
            "num_participants": len(deltas),
            "total_k_eff": float(total_k_eff),
            "global_z_norm": float(np.linalg.norm(self.global_z)),
        }
