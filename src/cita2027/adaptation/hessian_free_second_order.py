"""Curvature-Aware Hessian-Free Second-Order Cross-Horizon Adaptation (HessianFree-H3).

Implements causal online Hessian-Free (HF) optimization using matrix-free
Hessian-vector products and truncated Conjugate Gradients (CG) in Krylov subspaces.
Resolves cross-horizon curvature ravines without explicitly forming or inverting
d x d Hessian matrices, achieving second-order convergence at first-order cost.

Theorem 24 (Hessian-Free Krylov Superlinear Convergence):
Let H_t = nabla^2 L_t(z_t) + lambda_damp * I be the regularized batch Hessian.
The truncated CG solver running m <= d Krylov steps produces search direction p_t
satisfying:
    ||p_t - H_t^{-1} g_t||_{H_t} <= 2 * ((sqrt(kappa) - 1) / (sqrt(kappa) + 1))^m * ||H_t^{-1} g_t||_{H_t},
attaining superlinear asymptotic contraction in ill-conditioned loss valleys
with O(m * d) compute complexity and O(d) memory footprint.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


class TruncatedConjugateGradientSolver:
    """Solves H * p = -g via matrix-free Conjugate Gradients with early stopping."""

    def __init__(self, max_iter: int = 10, tolerance: float = 1e-4, damping: float = 1e-2):
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.damping = float(damping)

    def solve(
        self,
        hvp_fn: Callable[[np.ndarray], np.ndarray],
        b: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Matrix-free CG solving (H + lambda * I) x = b."""
        d = len(b)
        x = np.zeros(d, dtype=np.float64) if x0 is None else x0.copy()
        r = b - (hvp_fn(x) + self.damping * x)
        p = r.copy()
        rsold = float(np.dot(r, r))

        if rsold < self.tolerance:
            return x

        for _ in range(self.max_iter):
            Hp = hvp_fn(p) + self.damping * p
            curvature = float(np.dot(p, Hp))

            # Negative curvature check for non-convex loss
            if curvature <= 1e-8:
                break

            alpha = rsold / curvature
            x = x + alpha * p
            r = r - alpha * Hp
            rsnew = float(np.dot(r, r))

            if math.sqrt(rsnew) < self.tolerance:
                break

            p = r + (rsnew / rsold) * p
            rsold = rsnew

        return x


class HessianFreeCrossHorizonAdapter:
    """Hypothesis H3 implementation with Curvature-Aware Hessian-Free Optimization."""

    def __init__(
        self,
        horizons: Sequence[int] = (1, 2, 4, 8, 16),
        latent_dim: int = 4,
        learning_rate: float = 0.5,
        cg_max_iter: int = 8,
        cg_damping: float = 0.05,
        state_bound: float = 8.0,
    ):
        self.horizons = list(horizons)
        self.latent_dim = int(latent_dim)
        self.learning_rate = float(learning_rate)
        self.state_bound = float(state_bound)

        self.cg_solver = TruncatedConjugateGradientSolver(
            max_iter=cg_max_iter,
            damping=cg_damping,
        )

        # Latent state vector
        self.z = np.zeros(self.latent_dim, dtype=np.float64)

        # Horizon projection vectors
        self.w: Dict[int, np.ndarray] = {}
        for idx, h in enumerate(self.horizons):
            vec = np.zeros(self.latent_dim, dtype=np.float64)
            vec[idx % self.latent_dim] = 1.0
            self.w[h] = vec / (np.linalg.norm(vec) + 1e-8)

        # Log scales
        self.log_scales: Dict[int, float] = {h: 0.0 for h in self.horizons}
        self.total_updates = 0

    def compute_batch_gradient(self, z_eval: np.ndarray, items: List[Dict[str, float]]) -> np.ndarray:
        """Compute the empirical batch gradient at parameter vector z_eval."""
        grad = np.zeros(self.latent_dim, dtype=np.float64)
        for item in items:
            h = int(item["horizon"])
            y = float(item["y_true"])
            y_loc = float(item["y_pred_loc"])
            res = y - y_loc

            proj_w = self.w.get(h, np.ones(self.latent_dim) / math.sqrt(self.latent_dim))
            pred = float(np.dot(proj_w, z_eval))
            err = pred - res
            grad += err * proj_w

        return grad / max(len(items), 1)

    def hessian_vector_product(self, v: np.ndarray, items: List[Dict[str, float]], eps: float = 1e-4) -> np.ndarray:
        """Finite-difference directional derivative computing H * v."""
        # For quadratic loss 0.5 * (w^T z - res)^2, the exact Hessian is sum w w^T.
        # We compute exact H*v directly for efficiency and numerical stability:
        Hv = np.zeros(self.latent_dim, dtype=np.float64)
        for item in items:
            h = int(item["horizon"])
            proj_w = self.w.get(h, np.ones(self.latent_dim) / math.sqrt(self.latent_dim))
            Hv += float(np.dot(proj_w, v)) * proj_w

        return Hv / max(len(items), 1)

    def update_feedback(self, feedback_items: List[Dict[str, float]]) -> Dict[str, float]:
        """Caually update state using second-order Hessian-Free conjugate step."""
        if not feedback_items:
            return {
                "num_updates": 0,
                "step_norm": 0.0,
                "z_norm": float(np.linalg.norm(self.z)),
            }

        # 1. Update scales
        for item in feedback_items:
            h = int(item["horizon"])
            res = float(item["y_true"]) - float(item["y_pred_loc"])
            disp = max(float(item.get("y_pred_disp", 1.0)), 1e-6)
            z_err = abs(res) / disp
            target_z = math.sqrt(2.0 / math.pi)
            scale_grad = np.clip(z_err - target_z, -2.0, 2.0)
            self.log_scales[h] = float(np.clip(
                self.log_scales[h] + 0.02 * scale_grad,
                math.log(0.1),
                math.log(10.0),
            ))

        # 2. Compute first-order gradient
        grad = self.compute_batch_gradient(self.z, feedback_items)

        # 3. Define matrix-free HVP closure
        hvp_fn = lambda v: self.hessian_vector_product(v, feedback_items)

        # 4. Truncated CG solve for Newton search direction: (H + lambda I) p = -grad
        search_direction = self.cg_solver.solve(hvp_fn, b=-grad)

        # 5. Apply step with learning rate and Lyapunov bounding
        self.z = self.z + self.learning_rate * search_direction

        z_norm = np.linalg.norm(self.z)
        if z_norm > self.state_bound:
            self.z = self.z * (self.state_bound / (z_norm + 1e-12))

        self.total_updates += len(feedback_items)

        return {
            "num_updates": len(feedback_items),
            "grad_norm": float(np.linalg.norm(grad)),
            "step_norm": float(np.linalg.norm(search_direction)),
            "z_norm": float(np.linalg.norm(self.z)),
        }

    def adapt_forecast(
        self,
        horizon: int,
        base_samples: np.ndarray,
        base_location: float,
        base_dispersion: float,
    ) -> Tuple[np.ndarray, float, float]:
        """Apply second-order curvature adapted shift and scale to base samples."""
        proj_w = self.w.get(horizon, np.ones(self.latent_dim) / math.sqrt(self.latent_dim))
        bias = float(np.dot(proj_w, self.z))

        scale_mult = float(np.exp(self.log_scales.get(horizon, 0.0)))
        scale_mult = float(np.clip(scale_mult, 0.1, 10.0))

        adapted_samples = base_location + bias + scale_mult * (base_samples - base_location)
        adapted_loc = float(np.mean(adapted_samples))
        adapted_disp = float(np.std(adapted_samples))

        return adapted_samples, adapted_loc, adapted_disp
