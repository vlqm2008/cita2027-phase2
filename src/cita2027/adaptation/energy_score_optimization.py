"""
Direct Multivariate Energy Score Online Optimization (ES-H3).

Instead of minimizing decoupled squared error losses and post-hoc variance matching,
ES-H3 directly computes analytical subgradients of the strictly proper Multivariate
Energy Score (Gneiting & Raftery, 2007) with respect to the shared representation manifold z_t.

Energy Score Formulation:
    ES(Y_hat, y) = (1/M) * Sum_{m=1}^M ||y_hat^{(m)} - y||_2 - (1 / (2*M^2)) * Sum_{m,m'} ||y_hat^{(m)} - y_hat^{(m')}||_2

Analytical Subgradient with respect to shared state z_t:
    grad_z ES = (1/M) * Sum_{m=1}^M  W^T (y_hat^{(m)} - y) / (||y_hat^{(m)} - y||_2 + eps)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


class HypothesisESH3_DirectOptimizer:
    """
    Direct Multivariate Energy Score Online Adapter (ES-H3).
    """

    def __init__(
        self,
        horizons: List[int],
        latent_dim: int = 2,
        learning_rate: float = 0.06,
        age_decay_gamma: float = 0.08,
        overlap_discount_beta: float = 0.5,
        max_state_norm: float = 8.0,
        step_clip: float = 1.2,
    ):
        self.horizons = sorted(list(horizons))
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.age_decay_gamma = age_decay_gamma
        self.overlap_discount_beta = overlap_discount_beta
        self.max_state_norm = max_state_norm
        self.step_clip = step_clip

        # Dyadic basis matrix
        self.w: Dict[int, np.ndarray] = {}
        for h in self.horizons:
            v = np.array([(math.log2(max(1, h)) / 4.0) ** k for k in range(latent_dim)], dtype=np.float64)
            self.w[h] = v / (np.linalg.norm(v) + 1e-12)

        # Shared states for location and scale
        self.z_state = np.zeros(latent_dim, dtype=np.float64)
        self.z_scale_state = np.zeros(latent_dim, dtype=np.float64)
        self.local_bias = {h: 0.0 for h in self.horizons}
        self.local_log_scale = {h: 0.0 for h in self.horizons}

    def get_corrections(self, h: int) -> Tuple[float, float]:
        """Compute Delta mu and scale multiplier for horizon h."""
        w_h = self.w[h]
        delta_mu = float(np.dot(w_h, self.z_state) + self.local_bias[h])
        log_scale = float(np.dot(w_h, self.z_scale_state) + self.local_log_scale[h])
        log_scale = float(np.clip(log_scale, -1.6, 1.6))
        return delta_mu, float(math.exp(log_scale))

    def adapt_samples(
        self,
        base_samples: np.ndarray,
        base_location: float,
        h: int,
    ) -> np.ndarray:
        delta_mu, scale_mult = self.get_corrections(h)
        return (base_location + delta_mu) + scale_mult * (base_samples - base_location)

    def update_with_feedback(
        self,
        feedback_items: List[ForecastFeedbackItem],
        current_t: int,
    ) -> Dict[str, float]:
        """
        Execute prequential gradient descent using the analytical Energy Score subgradient.
        """
        if not feedback_items:
            return {"k_eff": 0.0, "grad_norm": 0.0}

        k = len(feedback_items)
        raw_weights = []
        for item in feedback_items:
            age = max(0, current_t - item.origin_t)
            raw_weights.append(math.exp(-self.age_decay_gamma * age))

        # Kish Effective Sample Size
        sum_w = sum(raw_weights)
        sum_sq = sum(w ** 2 for w in raw_weights)
        k_eff = (sum_w ** 2) / (sum_sq + 1e-8)
        overlap_factor = (k_eff / k) ** self.overlap_discount_beta
        eta_eff = self.learning_rate * overlap_factor

        # Construct sub-vectors over matured items
        # True values y_obs and ensemble samples
        num_samples = min(len(item.base_samples) for item in feedback_items)
        num_samples = max(10, num_samples)

        y_true_vec = np.array([item.y_true for item in feedback_items], dtype=np.float64)

        # Adapted ensemble matrix of shape (num_samples, k)
        adapted_matrix = np.zeros((num_samples, k), dtype=np.float64)
        W_sub = np.zeros((k, self.latent_dim), dtype=np.float64)

        for j, item in enumerate(feedback_items):
            h = item.horizon
            W_sub[j] = self.w[h]
            base_loc = getattr(item, "base_location", None)
            if base_loc is None:
                base_loc = float(np.median(item.base_samples)) if len(item.base_samples) > 0 else 0.0

            adapted_s = self.adapt_samples(item.base_samples[:num_samples], base_loc, h)
            adapted_matrix[:, j] = adapted_s

        # Compute Energy Score subgradient:
        # grad_z ES = (1 / M) * Sum_{m=1}^M  W^T (y_hat^{(m)} - y) / (||y_hat^{(m)} - y||_2 + eps)
        grad_z = np.zeros(self.latent_dim, dtype=np.float64)
        for m in range(num_samples):
            diff = adapted_matrix[m, :] - y_true_vec  # shape (k,)
            norm_diff = float(np.linalg.norm(diff)) + 1e-6
            unit_diff = diff / norm_diff
            # Project through basis matrix W_sub
            grad_z += np.dot(W_sub.T, unit_diff)

        grad_z /= float(num_samples)

        # Scale update based on empirical dispersion ratio
        grad_scale = np.zeros(self.latent_dim, dtype=np.float64)
        for j, item in enumerate(feedback_items):
            h = item.horizon
            res = abs(item.y_true - float(np.mean(adapted_matrix[:, j])))
            pred_std = float(np.std(adapted_matrix[:, j])) + 1e-4
            err_scale = (res / pred_std) - 1.0
            grad_scale += -np.clip(err_scale, -2.0, 2.0) * self.w[h]

        grad_scale /= float(k)

        # Apply bounded gradient step
        step_z = -eta_eff * grad_z
        norm_step = float(np.linalg.norm(step_z))
        if norm_step > self.step_clip:
            step_z = (step_z / norm_step) * self.step_clip

        self.z_state += step_z

        step_s = -eta_eff * 0.5 * grad_scale
        norm_s = float(np.linalg.norm(step_s))
        if norm_s > self.step_clip:
            step_s = (step_s / norm_s) * self.step_clip

        self.z_scale_state += step_s

        # Projection onto compact ball
        total_norm = float(np.linalg.norm(self.z_state))
        if total_norm > self.max_state_norm:
            self.z_state = (self.z_state / total_norm) * self.max_state_norm

        return {
            "k_eff": float(k_eff),
            "grad_norm": float(norm_step),
            "z_norm": float(np.linalg.norm(self.z_state)),
        }
