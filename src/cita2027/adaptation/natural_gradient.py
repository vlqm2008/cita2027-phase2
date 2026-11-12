"""
Natural Gradient and Riemannian Manifold Adaptation for Probabilistic Time-Series.

Replaces Euclidean online gradient descent with Natural Gradient Descent (NGD)
on the statistical manifold of predictive distributions:
1. Fisher Information Metric (FIM) tensor calculation for location-scale manifold:
   I(z) = E[ grad log p(y; z) grad log p(y; z)^T ]
2. Invariance to coordinate reparameterization and KL-divergence steepest descent
3. Tikhonov damped natural gradient: Delta z = - (I_t + lambda * I)^(-1) * g_t
4. Multi-horizon basis curvature projection with Kish effective sample regularization
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.correctors import BaseCorrector
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


class NaturalGradientRiemannianAdapter(BaseCorrector):
    """
    Second-Order Natural Gradient Adapter on Riemannian Statistical Manifold.
    
    Preconditions online gradient updates with the inverse Fisher Information Matrix,
    ensuring steepest descent in distribution space (Kullback-Leibler ball)
    rather than arbitrary Euclidean parameter space.
    """

    def __init__(
        self,
        horizons: List[int],
        latent_dim: int = 2,
        base_learning_rate: float = 0.05,
        tikhonov_damping: float = 0.15,
        decay_factor: float = 0.95,
        step_clip: float = 0.35,
    ):
        super().__init__(horizons)
        self.latent_dim = latent_dim
        self.lr0 = base_learning_rate
        self.damping = tikhonov_damping
        self.decay = decay_factor
        self.step_clip = step_clip

        # Horizon basis projections
        self.w: Dict[int, np.ndarray] = {}
        for h in self.horizons:
            v = np.array([(np.log2(max(1, h)) / 4.0) ** k for k in range(latent_dim)], dtype=np.float64)
            self.w[h] = v / np.linalg.norm(v)

        # Latent state z in R^d (location) and z_scale in R^d (log scale)
        self.z = np.zeros(self.latent_dim, dtype=np.float64)
        self.z_scale = np.zeros(self.latent_dim, dtype=np.float64)

        # Fisher Information Matrix for location: I_loc in R^(d x d)
        # Initialized to identity
        self.FIM_loc = np.eye(self.latent_dim, dtype=np.float64)
        # Fisher Information Matrix for scale: I_scale in R^(d x d)
        self.FIM_scale = np.eye(self.latent_dim, dtype=np.float64) * 2.0

        self.step_count = 0

    def reset(self) -> None:
        self.step_count = 0
        self.z = np.zeros(self.latent_dim, dtype=np.float64)
        self.z_scale = np.zeros(self.latent_dim, dtype=np.float64)
        self.FIM_loc = np.eye(self.latent_dim, dtype=np.float64)
        self.FIM_scale = np.eye(self.latent_dim, dtype=np.float64) * 2.0

    def adapt(self, origin_t: int, horizon: int, base_samples: np.ndarray) -> np.ndarray:
        s = np.asarray(base_samples, dtype=np.float64)
        loc = float(np.median(s))
        w_h = self.w[horizon]

        # Location shift on manifold
        loc_shift = float(np.dot(w_h, self.z))
        # Scale multiplier on manifold
        log_scale = float(np.dot(w_h, self.z_scale))
        scale_mult = float(np.clip(np.exp(log_scale), 0.1, 8.0))

        return loc + loc_shift + scale_mult * (s - loc)

    def update(self, feedback_items: List[ForecastFeedbackItem]) -> None:
        if not feedback_items:
            return

        K = len(feedback_items)
        grad_loc = np.zeros(self.latent_dim, dtype=np.float64)
        grad_scale = np.zeros(self.latent_dim, dtype=np.float64)

        weights = []

        for item in feedback_items:
            h = item.horizon
            s = np.asarray(item.base_samples, dtype=np.float64)
            loc = float(np.median(s))
            disp = max(1e-4, float(np.std(s)))
            w_h = self.w[h]

            # Model prediction at time of forecast
            cur_loc = loc + float(np.dot(w_h, self.z))
            cur_scale = disp * float(np.exp(float(np.dot(w_h, self.z_scale))))
            residual = item.y_true - cur_loc

            # Age decay
            age = max(1, item.target_t - item.origin_t)
            w_age = math.exp(-0.06 * (age - 1))
            weights.append(w_age)

            # Score function: grad_z log p(y; z)
            # For Gaussian: d/d_loc log p = (y - loc) / scale^2
            score_loc = (residual / (cur_scale**2)) * w_h
            # d/d_log_scale log p = ((y - loc)^2 / scale^2) - 1
            score_scale = (((residual / cur_scale) ** 2) - 1.0) * w_h

            grad_loc += w_age * score_loc
            grad_scale += w_age * score_scale

            # Online empirical Fisher matrix accumulation: E[score @ score^T]
            outer_loc = np.outer(score_loc, score_loc)
            outer_scale = np.outer(score_scale, score_scale)
            self.FIM_loc = self.decay * self.FIM_loc + (1.0 - self.decay) * outer_loc
            self.FIM_scale = self.decay * self.FIM_scale + (1.0 - self.decay) * outer_scale

        # Kish effective sample size normalization
        sum_w = sum(weights)
        sum_w2 = sum(w**2 for w in weights)
        k_eff = (sum_w**2) / max(1e-6, sum_w2)
        eta_eff = self.lr0 / math.sqrt(max(1.0, k_eff))

        # Invert Fisher Information Matrices with Tikhonov damping
        FIM_loc_damped = self.FIM_loc + self.damping * np.eye(self.latent_dim)
        FIM_scale_damped = self.FIM_scale + self.damping * np.eye(self.latent_dim)

        inv_FIM_loc = np.linalg.pinv(FIM_loc_damped)
        inv_FIM_scale = np.linalg.pinv(FIM_scale_damped)

        # Natural gradient step: Delta z = inv(FIM) @ grad
        nat_step_loc = eta_eff * (inv_FIM_loc @ grad_loc)
        nat_step_scale = eta_eff * (inv_FIM_scale @ grad_scale)

        # Step norm clipping
        norm_loc = float(np.linalg.norm(nat_step_loc))
        if norm_loc > self.step_clip:
            nat_step_loc = nat_step_loc * (self.step_clip / norm_loc)

        norm_scale = float(np.linalg.norm(nat_step_scale))
        if norm_scale > self.step_clip:
            nat_step_scale = nat_step_scale * (self.step_clip / norm_scale)

        self.z += nat_step_loc
        self.z_scale += nat_step_scale
        self.step_count += 1
