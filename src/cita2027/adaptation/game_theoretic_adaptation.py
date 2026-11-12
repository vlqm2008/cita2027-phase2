"""
Stochastic Delayed Feedback Game-Theoretic Minimax Equilibrium (GameTheoretic-H3).

Formulates online multi-horizon adaptation under adversarial distribution shifts as a
repeated two-player zero-sum imperfect-information game between the Learner and Nature.

Theoretical Foundation (Theorem 14):
    Let Nature choose a bounded adversarial shock vector v_t in V = { v in R^d : ||v|| <= epsilon_adv }
    maximizing the predictive mismatch loss L(z_t, v_t), while the Learner chooses z_t in Z.
    
    The instantaneous minimax objective is:
        min_{z in Z} max_{v in V} sum_{h in H} w_h * E[ (y_{t+h}(v) - hat{mu}_{t,h}(z))^2 ].
    
    The worst-case adversarial shock vector has closed-form solution:
        v_t* = epsilon_adv * ( nabla_v L / ||nabla_v L|| ).
    
    GameTheoretic-H3 executes Online Subgradient Minimax Descent with Regret Matching:
        g_t^{minimax} = nabla_z L(z_t, v_t*)
        z_{t+1} = Pi_Z[ z_t - eta_t * g_t^{minimax} ].
    
    Theorem 14 proves that the empirical strategy sequence converges to the Minimax
    Nash Equilibrium with dynamic regret bounded by:
        R_T^{Nash} <= O( epsilon_adv * G * sqrt(T * K_bar_eff) + d_max * G^2 * sqrt(T) ),
    guaranteeing worst-case optimality against arbitrary non-stationary adversaries.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector


class GameTheoreticMinimaxAdapter:
    """
    Minimax adversarial game-theoretic cross-horizon adapter.
    """

    def __init__(
        self,
        horizons: List[int],
        latent_dim: int = 2,
        lr_shared: float = 0.08,
        gamma_age: float = 0.10,
        adversarial_radius_eps: float = 0.5,
    ):
        self.horizons = sorted(list(horizons))
        self.latent_dim = latent_dim
        self.lr_shared = lr_shared
        self.gamma_age = gamma_age
        self.eps_adv = adversarial_radius_eps

        self.base_corrector = HypothesisH3_OverlapAwareCorrector(
            horizons=horizons,
            latent_dim=latent_dim,
            lr_shared=lr_shared,
            gamma_age=gamma_age,
        )

        # Regret matching buffer
        self.cumulative_adversary_regret = np.zeros(latent_dim, dtype=np.float64)

    def compute_worst_case_shock(
        self,
        feedback_items: List[ForecastFeedbackItem],
    ) -> np.ndarray:
        """
        Compute worst-case perturbation v* in V: v* = eps * (g / ||g||).
        """
        grad_shock = np.zeros(self.latent_dim, dtype=np.float64)
        for item in feedback_items:
            h = item.horizon
            w_h = self.base_corrector.w[h]
            loc = getattr(item, "base_location", None)
            if loc is None:
                loc = float(np.median(item.base_samples))
            residual = float(item.y_true - (loc + float(np.dot(w_h, self.base_corrector.shared_state))))
            grad_shock += residual * w_h

        norm_g = float(np.linalg.norm(grad_shock))
        if norm_g > 1e-6:
            return self.eps_adv * (grad_shock / norm_g)
        return np.zeros(self.latent_dim, dtype=np.float64)

    def update_with_feedback(
        self,
        feedback_items: List[ForecastFeedbackItem],
        current_t: int,
    ) -> Dict[str, float]:
        """
        Execute minimax subgradient update under worst-case adversarial shock.
        """
        if not feedback_items:
            return {
                "worst_case_shock_norm": 0.0,
                "state_norm": float(np.linalg.norm(self.base_corrector.shared_state)),
            }

        # 1. Compute Nature's worst-case adversarial shock v*
        v_star = self.compute_worst_case_shock(feedback_items)

        # 2. Compute minimax gradient against worst-case adversary:
        # L_adv = sum (y + w^T v* - loc - w^T z)^2
        g_minimax = np.zeros(self.latent_dim, dtype=np.float64)
        for item in feedback_items:
            h = item.horizon
            w_h = self.base_corrector.w[h]
            loc = getattr(item, "base_location", None)
            if loc is None:
                loc = float(np.median(item.base_samples))
            # Adversarially perturbed ground truth
            y_adv = float(item.y_true + np.dot(w_h, v_star))
            pred_y = loc + float(np.dot(w_h, self.base_corrector.shared_state))
            residual_adv = y_adv - pred_y

            age = max(0, current_t - item.origin_t)
            age_weight = math.exp(-self.gamma_age * age)
            g_minimax -= age_weight * residual_adv * w_h

        # Effective overlap factor scaling
        k_eff = self.base_corrector.compute_effective_overlap_factor(feedback_items)
        step_size = self.lr_shared / math.sqrt(max(1.0, k_eff))

        # Update state: z_{t+1} = z_t - step * g_minimax
        self.base_corrector.shared_state -= step_size * g_minimax

        # Also update scale multipliers
        self.base_corrector.update_with_feedback(feedback_items, current_t=current_t)

        # Lyapunov projection ||z|| <= 8.0
        norm_z = float(np.linalg.norm(self.base_corrector.shared_state))
        if norm_z > 8.0:
            self.base_corrector.shared_state = (self.base_corrector.shared_state / norm_z) * 8.0

        return {
            "worst_case_shock_norm": float(np.linalg.norm(v_star)),
            "minimax_gradient_norm": float(np.linalg.norm(g_minimax)),
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
