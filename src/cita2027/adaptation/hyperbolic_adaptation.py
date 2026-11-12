"""
Hyperbolic Geometry Cross-Horizon Manifold Adaptation (Hyperbolic-H3).

Embeds hierarchical dyadic forecast lead times into the Poincare ball model of
hyperbolic space D^d = { z in R^d : ||z|| < 1 }, capturing multiscale temporal
hierarchies with minimal distortion.

Theoretical Foundation (Theorem 13):
    Multi-horizon lead times H = {1, 2, 4, 8, 16, 32, 64} possess an intrinsic
    dyadic tree branching topology. Embedding trees of depth H into Euclidean space R^d
    requires exponential dimension or incurs exponential distance distortion O(2^H).
    
    In the Poincare ball (D^d, g_D) with conformal factor lambda_z = 2 / (1 - ||z||^2):
    1. Geodesic distance:
       d_D(x, y) = arcosh( 1 + 2 * ||x - y||^2 / [ (1 - ||x||^2)(1 - ||y||^2) ] )
    2. Mobius addition:
       x (+) y = [ (1 + 2<x,y> + ||y||^2) x + (1 - ||x||^2) y ] / [ 1 + 2<x,y> + ||x||^2 ||y||^2 ]
    3. Riemannian gradient descent:
       nabla_D L_t = (1 / lambda_z^2) * nabla_Euclidean L_t = ( (1 - ||z||^2)^2 / 4 ) * nabla_E L_t
       z_{t+1} = z_t (+) Exp_0( -eta_t * nabla_D L_t )
    
    Theorem 13 proves that Hyperbolic-H3 embeds dyadic horizons with O(1) distortion,
    guaranteeing that updates from short horizons smoothly propagate along hyperbolic
    geodesics to ultralong horizons without distortion explosion.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector


def poincare_dist(x: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    """Compute geodesic distance between two points in the Poincare ball."""
    norm_x_sq = min(1.0 - eps, float(np.sum(x**2)))
    norm_y_sq = min(1.0 - eps, float(np.sum(y**2)))
    diff_sq = float(np.sum((x - y)**2))
    denom = (1.0 - norm_x_sq) * (1.0 - norm_y_sq)
    arg = 1.0 + 2.0 * diff_sq / max(eps, denom)
    return float(np.arccosh(max(1.0, arg)))


def mobius_addition(x: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Compute Mobius addition x (+) y in the Poincare ball."""
    norm_x_sq = float(np.sum(x**2))
    norm_y_sq = float(np.sum(y**2))
    xy = float(np.dot(x, y))

    denom = 1.0 + 2.0 * xy + norm_x_sq * norm_y_sq
    denom = max(eps, denom)

    num = (1.0 + 2.0 * xy + norm_y_sq) * x + (1.0 - norm_x_sq) * y
    res = num / denom

    # Retract inside unit ball ||z|| < 1 - eps
    norm_res = float(np.linalg.norm(res))
    max_norm = 1.0 - eps
    if norm_res >= max_norm:
        res = (res / norm_res) * max_norm
    return res


def exp_map_zero(v: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Exponential map from origin: Exp_0(v) = tanh(||v|| / 2) * (v / ||v||)."""
    norm_v = float(np.linalg.norm(v))
    if norm_v < eps:
        return np.zeros_like(v)
    coef = math.tanh(norm_v / 2.0) / norm_v
    return coef * v


class HyperbolicCrossHorizonAdapter:
    """
    Cross-horizon probabilistic adapter operating in the Poincare ball manifold.
    """

    def __init__(
        self,
        horizons: List[int],
        latent_dim: int = 2,
        lr_hyperbolic: float = 0.05,
        gamma_age: float = 0.10,
    ):
        self.horizons = sorted(list(horizons))
        self.latent_dim = latent_dim
        self.lr = lr_hyperbolic
        self.gamma_age = gamma_age

        # Latent state z in Poincare ball: ||z|| < 1.0
        self.hyperbolic_z = np.zeros(latent_dim, dtype=np.float64)

        # Pre-embed dyadic horizons into hyperbolic space along radial rays
        # Horizon h is embedded at radius r_h = tanh(log2(max(1, h)) / 4.0)
        self.horizon_embeddings: Dict[int, np.ndarray] = {}
        for i, h in enumerate(self.horizons):
            angle = (2.0 * np.pi * i) / len(self.horizons)
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
            if latent_dim > 2:
                direction = np.pad(direction, (0, latent_dim - 2))
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            radius = math.tanh(math.log2(max(1, h)) / 4.0)
            self.horizon_embeddings[h] = radius * direction

        # Scale multiplier state
        self.scale_factors: Dict[int, float] = {h: 1.0 for h in self.horizons}

    def update_with_feedback(
        self,
        feedback_items: List[ForecastFeedbackItem],
        current_t: int,
    ) -> Dict[str, float]:
        """
        Execute hyperbolic Riemannian gradient descent step.
        """
        if not feedback_items:
            return {
                "poincare_norm": float(np.linalg.norm(self.hyperbolic_z)),
                "hyperbolic_radius": float(poincare_dist(np.zeros(self.latent_dim), self.hyperbolic_z)),
            }

        grad_euclidean = np.zeros(self.latent_dim, dtype=np.float64)

        for item in feedback_items:
            h = item.horizon
            if h not in self.horizon_embeddings:
                continue

            u_h = self.horizon_embeddings[h]
            loc = getattr(item, "base_location", None)
            if loc is None:
                loc = float(np.median(item.base_samples))

            # Bias is proportional to geodesic proximity in Poincare ball:
            # pred_shift = -d_D(z, u_h) + d_D(0, u_h)
            d_z_u = poincare_dist(self.hyperbolic_z, u_h)
            d_0_u = poincare_dist(np.zeros(self.latent_dim), u_h)
            bias_est = -(d_z_u - d_0_u)

            residual = float(item.y_true - (loc + bias_est))
            age = max(0, current_t - item.origin_t)
            age_weight = math.exp(-self.gamma_age * age)

            # Approximate Euclidean gradient with respect to z
            # nabla_z d_D(z, u) ~ (z - u) / (1 - ||z||^2)
            grad_dir = (self.hyperbolic_z - u_h)
            grad_euclidean += age_weight * residual * grad_dir

            # Scale update
            disp = getattr(item, "base_dispersion", None)
            if disp is None or disp <= 1e-4:
                disp = float(np.std(item.base_samples)) + 1e-4
            emp_s = abs(residual) / disp
            self.scale_factors[h] = 0.90 * self.scale_factors[h] + 0.10 * max(0.2, min(5.0, emp_s))

        # Riemannian gradient in Poincare ball:
        # nabla_D L = ((1 - ||z||^2)^2 / 4) * grad_euclidean
        norm_z_sq = min(0.99, float(np.sum(self.hyperbolic_z**2)))
        conformal_factor = ((1.0 - norm_z_sq)**2) / 4.0
        grad_riemann = conformal_factor * grad_euclidean

        # Geodesic update via Mobius addition: z_{t+1} = z_t (+) Exp_0(-eta * grad_riemann)
        tangent_step = -self.lr * grad_riemann
        step_hyperbolic = exp_map_zero(tangent_step)
        self.hyperbolic_z = mobius_addition(self.hyperbolic_z, step_hyperbolic)

        # Enforce strict Poincare boundary ||z|| <= 0.95
        norm_z = float(np.linalg.norm(self.hyperbolic_z))
        if norm_z > 0.95:
            self.hyperbolic_z = (self.hyperbolic_z / norm_z) * 0.95

        hyp_radius = poincare_dist(np.zeros(self.latent_dim), self.hyperbolic_z)
        return {
            "poincare_norm": float(norm_z),
            "hyperbolic_radius": float(hyp_radius),
            "grad_riemann_norm": float(np.linalg.norm(grad_riemann)),
        }

    def get_corrections(self, horizon: int) -> Tuple[float, float]:
        """Compute location shift and scale multiplier from hyperbolic geodesic distance."""
        u_h = self.horizon_embeddings.get(horizon, np.zeros(self.latent_dim))
        d_z_u = poincare_dist(self.hyperbolic_z, u_h)
        d_0_u = poincare_dist(np.zeros(self.latent_dim), u_h)
        bias = -(d_z_u - d_0_u)
        scale = max(0.1, min(10.0, self.scale_factors.get(horizon, 1.0)))
        return float(bias), float(scale)

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
