"""Spatio-Temporal Cross-Horizon Adaptation with Graph Laplacian Regularization (ST-H3).

Extends asynchronous cross-horizon adaptation to multi-node spatial networks (e.g., power grids,
traffic sensor networks, multi-station weather sensors).

Mathematical Formulation:
- Spatial network: G = (V, E, A) with N nodes and normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
- Latent state matrix: Z_t = [z_{t, 1}, ..., z_{t, N}]^T in R^{N x d}.
- Joint Spatio-Temporal Adaptation Objective:
    min_{Z} sum_{i=1}^N ell_i(Z_i) + lambda_G * Tr(Z^T L Z)
- Update Rule under Asynchronous Arrival at node i:
    z_{t+1, i} = z_{t, i} + eta_shared * (g_{t, i} - lambda_G * [L Z_t]_i) / sqrt(K_eff)
  where spatial diffusion [L Z_t]_i instantly informs neighboring nodes j in N(i)
  of localized structural shocks observed at node i before node j receives its own feedback!
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SpatialFeedbackItem:
    """Feedback item carrying node ID, origin, horizon, target, and values."""
    node_id: int
    origin_t: int
    horizon: int
    target_t: int
    y_true: float
    base_samples: np.ndarray


class SpatioTemporalCrossHorizonAdapter:
    """ST-H3: Graph-Laplacian regularized multi-node cross-horizon adapter."""

    def __init__(
        self,
        num_nodes: int,
        horizons: List[int],
        adjacency_matrix: Optional[np.ndarray] = None,
        latent_dim: int = 2,
        lr_shared: float = 0.08,
        lr_spatial: float = 0.04,
        lambda_graph: float = 0.15,
        gamma_age: float = 0.12,
        overlap_discount_factor: float = 0.70,
    ) -> None:
        self.num_nodes = num_nodes
        self.horizons = sorted(horizons)
        self.latent_dim = latent_dim
        self.lr_shared = lr_shared
        self.lr_spatial = lr_spatial
        self.lambda_graph = lambda_graph
        self.gamma_age = gamma_age
        self.overlap_discount_factor = overlap_discount_factor

        # Spatial Graph Adjacency and Laplacian
        if adjacency_matrix is None:
            # Default to ring lattice or line topology
            A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if abs(i - j) == 1 or abs(i - j) == num_nodes - 1:
                        A[i, j] = 1.0
            self.A = A
        else:
            self.A = np.asarray(adjacency_matrix, dtype=np.float64)

        # Compute Normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
        deg = np.sum(self.A, axis=1)
        deg_inv_sqrt = np.zeros_like(deg)
        mask = deg > 1e-8
        deg_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        self.L = np.eye(num_nodes) - D_inv_sqrt @ self.A @ D_inv_sqrt

        # Multi-horizon projection basis w_h
        self.w: Dict[int, np.ndarray] = {}
        for h in self.horizons:
            v = np.array([(math.log2(max(1, h)) / 4.0) ** k for k in range(latent_dim)], dtype=np.float64)
            self.w[h] = v / np.linalg.norm(v)

        # Latent state matrix Z in R^{N x d}
        self.Z = np.zeros((num_nodes, latent_dim), dtype=np.float64)
        # Local bias per (node, horizon)
        self.local_bias: Dict[Tuple[int, int], float] = {
            (i, h): 0.0 for i in range(num_nodes) for h in self.horizons
        }

    def adapt(
        self,
        node_id: int,
        origin_t: int,
        horizon: int,
        base_samples: np.ndarray,
    ) -> np.ndarray:
        """Adapts predictive samples for specific node and horizon."""
        s = np.asarray(base_samples, dtype=np.float64)
        loc = float(np.median(s))

        # Shared projection from node's latent state Z[node_id]
        shared_bias = float(np.dot(self.w[horizon], self.Z[node_id]))
        loc_b = self.local_bias[(node_id, horizon)]
        total_bias = shared_bias + loc_b
        return loc + total_bias + (s - loc)

    def update(self, feedback_items: List[SpatialFeedbackItem]) -> None:
        """Spatio-temporal graph update using multi-node feedback batch."""
        if not feedback_items and np.allclose(self.Z, 0.0):
            return

        # 1. Group items by node to compute effective overlap per node
        items_by_node: Dict[int, List[SpatialFeedbackItem]] = {i: [] for i in range(self.num_nodes)}
        for item in feedback_items:
            items_by_node[item.node_id].append(item)

        # 2. Compute spatial graph diffusion gradient: grad_graph = L @ Z
        spatial_laplacian_diff = self.L @ self.Z  # shape (N, d)

        # 3. Update each node
        for i in range(self.num_nodes):
            n_items = items_by_node[i]
            if not n_items:
                # Even if node i has no feedback, diffuse information from neighbors!
                self.Z[i] -= self.lr_spatial * self.lambda_graph * spatial_laplacian_diff[i]
                continue

            k = len(n_items)
            # Compute effective overlap factor
            total_overlap = 0.0
            for a in range(k):
                for b in range(a + 1, k):
                    ratio = min(n_items[a].horizon, n_items[b].horizon) / max(n_items[a].horizon, n_items[b].horizon)
                    total_overlap += 2.0 * ratio * self.overlap_discount_factor
            k_eff = max(1.0, float(k + total_overlap))
            overlap_scale = 1.0 / math.sqrt(k_eff)

            node_grad = np.zeros(self.latent_dim, dtype=np.float64)
            for item in n_items:
                h = item.horizon
                age = float(h)
                age_weight = math.exp(-self.gamma_age * age)

                s = np.asarray(item.base_samples, dtype=np.float64)
                loc = float(np.median(s))
                err = item.y_true - loc

                # Project error onto horizon basis
                node_grad += age_weight * err * self.w[h]
                # Update local bias
                self.local_bias[(i, h)] += 0.03 * age_weight * err

            # Combined update: local observation gradient + graph Laplacian spatial coupling
            net_gradient = (node_grad * overlap_scale) - (self.lambda_graph * spatial_laplacian_diff[i])
            self.Z[i] += self.lr_shared * net_gradient

            # Clip norm for numerical stability
            norm = np.linalg.norm(self.Z[i])
            if norm > 8.0:
                self.Z[i] = (self.Z[i] / norm) * 8.0

    def reset(self) -> None:
        self.Z.fill(0.0)
        self.local_bias = {
            (i, h): 0.0 for i in range(self.num_nodes) for h in self.horizons
        }
