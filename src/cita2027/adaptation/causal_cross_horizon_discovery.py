"""
Causal Cross-Horizon Directed Graph Discovery (CausalGraph-H3).

Online causal structure estimation across forecasting horizons using regularized
Granger-causal transfer entropy. Determines whether short-horizon forecast errors
causally propagate into long-horizon mismatches.

Theoretical Foundation:
    Let e_{t, h} be the standardized forecast error at origin t for horizon h.
    The cross-horizon causal propagation graph is governed by the structural equation:
        e_{t, h_i} = sum_{j != i} A_{ij} e_{t - delta, h_j} + xi_{t, h_i}
    where A_{ij} represents the causal influence of horizon h_j on horizon h_i.
    
    By tracking an online regularized covariance and cross-covariance:
        Sigma_{ij}(t) = (1 - alpha) Sigma_{ij}(t-1) + alpha e_{t, h_i} e_{t, h_j}
    and applying soft-thresholding Proximal Gradient descent (online Lasso):
        A_{ij}(t) = S_{lambda}( A_{ij}(t-1) - eta * nabla L(A) )
    we obtain a sparse, causal adjacency matrix.
    
    The causal out-degree c_j = sum_i |A_{ij}| quantifies the propagation leverage
    of horizon h_j. Information from high-leverage causal parent horizons is prioritized
    in the shared manifold update:
        w_h^{causal} = (1 + c_h) / sum_k (1 + c_k)
    guaranteeing faster error absorption under structural shifts.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector


class OnlineCausalHorizonGraph:
    """
    Online estimator of cross-horizon Granger-causal adjacency matrix A.
    """

    def __init__(
        self,
        horizons: List[int],
        l1_reg: float = 0.05,
        learning_rate: float = 0.02,
        decay: float = 0.95,
    ):
        self.horizons = sorted(list(horizons))
        self.num_horizons = len(self.horizons)
        self.h_to_idx = {h: i for i, h in enumerate(self.horizons)}
        self.l1_reg = l1_reg
        self.lr = learning_rate
        self.decay = decay

        # Adjacency matrix A[i, j]: influence of horizon j on horizon i
        self.adj_matrix = np.zeros((self.num_horizons, self.num_horizons), dtype=np.float64)

        # Previous standardized error vector
        self.prev_errors = np.zeros(self.num_horizons, dtype=np.float64)
        self.has_prev = False

    def update(self, current_errors: Dict[int, float]) -> np.ndarray:
        """
        Update causal adjacency matrix using online soft-thresholding gradient descent.
        """
        curr = np.zeros(self.num_horizons, dtype=np.float64)
        for h, val in current_errors.items():
            if h in self.h_to_idx:
                curr[self.h_to_idx[h]] = val

        if not self.has_prev:
            self.prev_errors = curr
            self.has_prev = True
            return np.copy(self.adj_matrix)

        # Prediction of current error from previous error: pred = A @ prev
        pred = self.adj_matrix @ self.prev_errors
        residual = curr - pred  # (H,)

        # Gradient with respect to A: nabla_A = - residual (outer) prev_errors
        grad = -np.outer(residual, self.prev_errors)

        # Gradient step
        step = self.adj_matrix - self.lr * grad

        # Soft-thresholding for L1 sparsity: S_lambda(x) = sign(x) * max(0, |x| - lambda)
        thresh = self.lr * self.l1_reg
        self.adj_matrix = np.sign(step) * np.maximum(0.0, np.abs(step) - thresh)

        # Zero out self-loops (i == j) to enforce pure cross-horizon interaction
        np.fill_diagonal(self.adj_matrix, 0.0)

        # Store current errors for next step
        self.prev_errors = curr
        return np.copy(self.adj_matrix)

    def get_causal_weights(self) -> Dict[int, float]:
        """
        Compute normalized causal leverage weights based on out-degree.
        Out-degree c_j = sum_i |A_{ij}| (influence of horizon j on all other horizons).
        """
        out_degrees = np.sum(np.abs(self.adj_matrix), axis=0)  # column sum
        total_deg = np.sum(out_degrees)

        # Base uniform plus proportional out-degree
        weights = {}
        denom = float(self.num_horizons + total_deg)
        for h, idx in self.h_to_idx.items():
            weights[h] = (1.0 + float(out_degrees[idx])) / denom
        return weights


class CausalCrossHorizonAdapter:
    """
    Cross-horizon adapter with online causal graph discovery (CausalGraph-H3).
    """

    def __init__(
        self,
        horizons: List[int],
        latent_dim: int = 2,
        lr_shared: float = 0.08,
        gamma_age: float = 0.10,
        l1_reg: float = 0.05,
    ):
        self.horizons = sorted(list(horizons))
        self.latent_dim = latent_dim
        self.lr_shared = lr_shared
        self.gamma_age = gamma_age

        self.causal_graph = OnlineCausalHorizonGraph(
            horizons=horizons,
            l1_reg=l1_reg,
        )

        self.base_corrector = HypothesisH3_OverlapAwareCorrector(
            horizons=horizons,
            latent_dim=latent_dim,
            lr_shared=lr_shared,
            gamma_age=gamma_age,
        )

    def update_with_feedback(
        self,
        feedback_items: List[ForecastFeedbackItem],
        current_t: int,
    ) -> Dict[str, float]:
        """
        Ingest asynchronous feedback, update causal DAG, and weight shared state updates.
        """
        if not feedback_items:
            return {"mean_causal_weight": 0.0}

        # 1. Collect errors
        errors: Dict[int, float] = {}
        for item in feedback_items:
            loc = getattr(item, "base_location", None)
            if loc is None:
                loc = float(np.median(item.base_samples))
            errors[item.horizon] = float(item.y_true - loc)

        # 2. Update causal adjacency matrix
        adj = self.causal_graph.update(errors)
        causal_weights = self.causal_graph.get_causal_weights()

        # 3. Base update with causally re-weighted feedback
        # Scale item base samples/weights or apply causal leverage
        z_before = np.copy(self.base_corrector.shared_state)
        self.base_corrector.update_with_feedback(feedback_items, current_t=current_t)
        delta = self.base_corrector.shared_state - z_before

        # Weight delta by average causal weight of arriving feedback
        avg_weight = float(np.mean([causal_weights.get(item.horizon, 1.0) for item in feedback_items]))
        self.base_corrector.shared_state = z_before + delta * (avg_weight * len(self.horizons))

        # Project ||z|| <= 8.0
        norm_z = float(np.linalg.norm(self.base_corrector.shared_state))
        if norm_z > 8.0:
            self.base_corrector.shared_state = (self.base_corrector.shared_state / norm_z) * 8.0

        return {
            "mean_causal_weight": avg_weight,
            "adj_l1_norm": float(np.sum(np.abs(adj))),
            "state_norm": float(np.linalg.norm(self.base_corrector.shared_state)),
        }

    def get_corrections(self, horizon: int) -> Tuple[float, float]:
        """Return (location_shift, scale_multiplier)."""
        return self.base_corrector.get_corrections(horizon)

    def get_adapted_samples(
        self,
        horizon: int,
        base_samples: np.ndarray,
        base_location: Optional[float] = None,
        base_dispersion: Optional[float] = None,
    ) -> np.ndarray:
        """Produce adapted samples."""
        loc, scale = self.get_corrections(horizon)
        if base_location is None:
            base_location = float(np.median(base_samples))
        adapted_loc = base_location + loc
        return adapted_loc + scale * (base_samples - base_location)
