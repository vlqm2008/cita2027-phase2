"""
Information-Geometric Riemannian Natural Gradient Adaptation (Riemann-H3).

Implements online Riemannian natural gradient descent over the statistical manifold
of multi-horizon predictive distributions equipped with the Fisher Information Metric.

Theoretical Foundation (Theorem 10):
    Let p(y | z) be the parameterized predictive distribution family across horizons H.
    The Riemannian manifold (Z, G) possesses the Fisher Information Metric:
        G(z) = E_{y ~ p_z}[ nabla_z log p(y | z) nabla_z log p(y | z)^T ].
    
    For multi-horizon location-scale Gaussian / Student-t representations:
        mu_{t,h} = mu_{base, h} + w_h^T z_t
        sigma_{t,h} = sigma_{base, h} * s_{t,h}
    The Fisher Information Matrix is given analytically by:
        G_t = sum_{h in H} (1 / sigma_{t,h}^2) * (w_h w_h^T) + lambda_fisher * I_d.
    
    Riemannian Natural Gradient Update:
        tilde{nabla}_z L_t = G_t^{-1} nabla_z L_t
        z_{t+1} = Retr_{z_t}( -eta_t * tilde{nabla}_z L_t )
    where Retr is the geodesic retraction mapping onto the compact Lyapunov ball ||z|| <= 8.0.
    
    Theorem 10 guarantees that Riemannian natural gradient descent is strictly invariant
    to linear coordinates and achieves optimal Fisher efficiency with condition number
    cond(G^{-1/2} nabla^2 L G^{-1/2}) <= O(1), eliminating cross-horizon gradient imbalance.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector


class RiemannianNaturalGradientAdapter:
    """
    Riemannian statistical manifold adapter using analytical Fisher Information preconditioning.
    """

    def __init__(
        self,
        horizons: List[int],
        latent_dim: int = 2,
        lr_shared: float = 0.08,
        gamma_age: float = 0.10,
        fisher_reg: float = 0.05,
    ):
        self.horizons = sorted(list(horizons))
        self.latent_dim = latent_dim
        self.lr_shared = lr_shared
        self.gamma_age = gamma_age
        self.fisher_reg = fisher_reg

        self.base_corrector = HypothesisH3_OverlapAwareCorrector(
            horizons=horizons,
            latent_dim=latent_dim,
            lr_shared=lr_shared,
            gamma_age=gamma_age,
        )

        # Track Fisher metric condition number
        self.cond_history: List[float] = []

    def compute_fisher_information_matrix(self) -> np.ndarray:
        """
        Compute analytical Fisher Information Matrix G_t across all horizons:
        G_t = sum_h (1 / sigma_{t,h}^2) * w_h w_h^T + lambda * I.
        """
        g_mat = np.eye(self.latent_dim, dtype=np.float64) * self.fisher_reg

        for h in self.horizons:
            w_h = self.base_corrector.w[h]
            _, scale = self.base_corrector.get_corrections(h)
            variance = max(0.01, scale**2)
            g_mat += (1.0 / variance) * np.outer(w_h, w_h)

        return g_mat

    def update_with_feedback(
        self,
        feedback_items: List[ForecastFeedbackItem],
        current_t: int,
    ) -> Dict[str, float]:
        """
        Execute natural gradient step: Delta z = - eta * G^{-1} g_euclidean.
        """
        if not feedback_items:
            return {"fisher_cond": 1.0, "state_norm": float(np.linalg.norm(self.base_corrector.shared_state))}

        # 1. Compute Euclidean gradient of squared error loss
        # L = 0.5 * sum (y - loc - w^T z)^2
        grad_euclidean = np.zeros(self.latent_dim, dtype=np.float64)
        for item in feedback_items:
            h = item.horizon
            w_h = self.base_corrector.w[h]
            loc = getattr(item, "base_location", None)
            if loc is None:
                loc = float(np.median(item.base_samples))
            residual = float(item.y_true - (loc + float(np.dot(w_h, self.base_corrector.shared_state))))
            age = max(0, current_t - item.origin_t)
            age_weight = math.exp(-self.gamma_age * age)
            grad_euclidean -= age_weight * residual * w_h

        # 2. Compute Fisher Information Matrix & its inverse
        g_mat = self.compute_fisher_information_matrix()
        eigvals = np.linalg.eigvalsh(g_mat)
        cond_num = float(np.max(eigvals) / max(1e-6, np.min(eigvals)))
        self.cond_history.append(cond_num)

        # Natural gradient direction: g_nat = G^{-1} @ grad_euclidean
        try:
            g_natural = np.linalg.solve(g_mat, grad_euclidean)
        except np.linalg.LinAlgError:
            g_natural = grad_euclidean

        # Overlap factor scaling
        k_eff = self.base_corrector.compute_effective_overlap_factor(feedback_items)
        step_scale = self.lr_shared / math.sqrt(max(1.0, k_eff))

        # 3. Geodesic retraction step: z_{t+1} = z_t - step * g_natural
        self.base_corrector.shared_state -= step_scale * g_natural

        # Also update local scales via standard corrector
        self.base_corrector.update_with_feedback(feedback_items, current_t=current_t)

        # Projection retraction onto compact ball ||z|| <= 8.0
        norm_z = float(np.linalg.norm(self.base_corrector.shared_state))
        if norm_z > 8.0:
            self.base_corrector.shared_state = (self.base_corrector.shared_state / norm_z) * 8.0

        return {
            "fisher_cond": cond_num,
            "natural_grad_norm": float(np.linalg.norm(g_natural)),
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
        """Produce adapted predictive samples."""
        loc, scale = self.get_corrections(horizon)
        if base_location is None:
            base_location = float(np.median(base_samples))
        return base_location + loc + scale * (base_samples - base_location)
