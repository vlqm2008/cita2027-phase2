"""
Jump-Diffusion Decomposition for Asynchronous Cross-Horizon Adaptation (JumpDiff-H3).

Decouples continuous diffusive drift from discrete Poisson jump shocks in
multi-horizon time-series foundation model residuals.

Theoretical Foundation (Theorem 7):
    Let residual increments follow a jump-diffusion process:
        d epsilon_t = mu_t dt + sigma_t dW_t + J_t dN_t
    where N_t is a Poisson jump process with intensity lambda_t, and J_t ~ N(mu_J, sigma_J^2).
    Standard online gradient updates conflate continuous volatility sigma_t^2 with jump variance.
    By estimating Realized Bipower Variation (BV) and Realized Variance (RV):
        BV_t = (pi/2) sum |r_i| |r_{i-1}| -> converges to continuous diffusion sigma_t^2 dt
        RV_t = sum r_i^2 -> converges to sigma_t^2 dt + sum J_t^2
    Jump intensity is identified as J_ratio = max(0, (RV - BV) / RV).
    
    Under JumpDiff-H3:
    1. Manifold state update is robustly shielded against jump outliers:
       Delta z_t = Delta z_t / (1 + kappa_jump * J_ratio)
    2. Predictive scale multiplier incorporates horizon-scaled jump cushion:
       s_{t,h} = s_{base} * sqrt(1 + beta_jump * J_ratio * (h / H_max))
    guaranteeing bounded Lyapunov drift ||z_t|| <= 8.0 and strictly positive scale s_{t,h} > 0.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector


class JumpDiffusionOnlineFilter:
    """
    Causal online estimator of continuous bipower variation and discrete jump intensity.
    """

    def __init__(
        self,
        window_size: int = 15,
        jump_threshold: float = 0.25,
        decay_factor: float = 0.92,
    ):
        self.window_size = window_size
        self.jump_threshold = jump_threshold
        self.decay = decay_factor

        self.residual_history: List[float] = []
        self.continuous_vol_est: float = 1.0
        self.jump_intensity_est: float = 0.0

    def observe_residual(self, error: float) -> Tuple[float, float, bool]:
        """
        Ingest newly arrived residual error and update diffusion/jump estimates causally.
        Returns: (continuous_vol, jump_intensity_ratio, is_jump_detected)
        """
        self.residual_history.append(float(error))
        if len(self.residual_history) > self.window_size * 3:
            self.residual_history.pop(0)

        if len(self.residual_history) < 3:
            return 1.0, 0.0, False

        arr = np.array(self.residual_history[-self.window_size :], dtype=np.float64)
        n = len(arr)
        if n < 3:
            return 1.0, 0.0, False

        # Realized Variance
        rv = float(np.sum(arr**2) / n)

        # Realized Bipower Variation: (pi / 2) * sum_{i=1}^{n-1} |r_i| * |r_{i-1}| / (n - 1)
        abs_r = np.abs(arr)
        bv = float((np.pi / 2.0) * np.sum(abs_r[1:] * abs_r[:-1]) / (n - 1))

        # Jump intensity fraction
        eps = 1e-8
        if rv > bv:
            raw_jump = (rv - bv) / (rv + eps)
        else:
            raw_jump = 0.0

        # EWMA causal smoothing
        self.continuous_vol_est = self.decay * self.continuous_vol_est + (1.0 - self.decay) * math.sqrt(max(bv, eps))
        self.jump_intensity_est = self.decay * self.jump_intensity_est + (1.0 - self.decay) * raw_jump

        is_jump = self.jump_intensity_est > self.jump_threshold
        return self.continuous_vol_est, self.jump_intensity_est, is_jump


class JumpDiffusionCrossHorizonAdapter:
    """
    Cross-Horizon Adapter equipped with Bipower Jump-Diffusion Filtering (JumpDiff-H3).
    """

    def __init__(
        self,
        horizons: List[int],
        latent_dim: int = 2,
        lr_shared: float = 0.08,
        gamma_age: float = 0.10,
        jump_damping_kappa: float = 2.0,
        jump_cushion_beta: float = 1.5,
    ):
        self.horizons = sorted(list(horizons))
        self.max_horizon = max(self.horizons)
        self.latent_dim = latent_dim
        self.lr_shared = lr_shared
        self.gamma_age = gamma_age
        self.kappa_jump = jump_damping_kappa
        self.beta_jump = jump_cushion_beta

        # Base overlap-aware corrector
        self.base_corrector = HypothesisH3_OverlapAwareCorrector(
            horizons=horizons,
            latent_dim=latent_dim,
            lr_shared=lr_shared,
            gamma_age=gamma_age,
        )

        # Horizon-specific jump filters
        self.filters: Dict[int, JumpDiffusionOnlineFilter] = {
            h: JumpDiffusionOnlineFilter() for h in self.horizons
        }

        # Current jump intensity state across horizons
        self.current_jump_intensities: Dict[int, float] = {h: 0.0 for h in self.horizons}

    def update_with_feedback(
        self,
        feedback_items: List[ForecastFeedbackItem],
        current_t: int,
    ) -> Dict[str, float]:
        """
        Process asynchronous feedback with jump detection and damped gradient updates.
        """
        if not feedback_items:
            return {"mean_jump_intensity": 0.0, "jump_detections": 0}

        # 1. Update jump filters for incoming items
        jump_count = 0
        total_jump_val = 0.0

        for item in feedback_items:
            h = item.horizon
            if h not in self.filters:
                continue
            loc = getattr(item, "base_location", None)
            if loc is None:
                loc = float(np.median(item.base_samples))
            residual = float(item.y_true - loc)

            _, j_val, is_jump = self.filters[h].observe_residual(residual)
            self.current_jump_intensities[h] = j_val
            total_jump_val += j_val
            if is_jump:
                jump_count += 1

        avg_jump = total_jump_val / len(feedback_items)

        # 2. Compute standard overlap-aware gradient update
        z_before = np.copy(self.base_corrector.shared_state)
        self.base_corrector.update_with_feedback(feedback_items, current_t=current_t)
        raw_delta = self.base_corrector.shared_state - z_before

        # 3. Apply jump damping shielding to prevent overshooting under discontinuous jumps
        damping = 1.0 / (1.0 + self.kappa_jump * avg_jump)
        damped_delta = raw_delta * damping

        # Update state with damped delta
        self.base_corrector.shared_state = z_before + damped_delta

        # Lyapunov projection ||z|| <= 8.0
        norm_z = float(np.linalg.norm(self.base_corrector.shared_state))
        if norm_z > 8.0:
            self.base_corrector.shared_state = (self.base_corrector.shared_state / norm_z) * 8.0

        return {
            "mean_jump_intensity": float(avg_jump),
            "jump_detections": int(jump_count),
            "damping_applied": float(damping),
            "state_norm": float(np.linalg.norm(self.base_corrector.shared_state)),
        }

    def get_adapted_samples(
        self,
        horizon: int,
        base_samples: np.ndarray,
        base_location: Optional[float] = None,
        base_dispersion: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate adapted predictive samples with jump cushion dispersion scaling.
        """
        loc, scale = self.base_corrector.get_corrections(horizon)

        if base_location is None:
            base_location = float(np.median(base_samples))

        # Base location shift
        adapted_loc = base_location + loc

        # Apply jump cushion to scale multiplier: s_adapted = s_base * sqrt(1 + beta * J * (h/H_max))
        j_val = self.current_jump_intensities.get(horizon, 0.0)
        h_ratio = float(horizon) / float(self.max_horizon)
        jump_cushion = math.sqrt(1.0 + self.beta_jump * j_val * h_ratio)
        effective_scale = max(0.1, min(10.0, scale * jump_cushion))

        # Adapted samples
        adapted_samples = adapted_loc + effective_scale * (base_samples - base_location)
        return adapted_samples

    def get_corrections(self, horizon: int) -> Tuple[float, float]:
        """Return (location_shift, scale_multiplier) including jump cushion."""
        loc, scale = self.base_corrector.get_corrections(horizon)
        j_val = self.current_jump_intensities.get(horizon, 0.0)
        h_ratio = float(horizon) / float(self.max_horizon)
        jump_cushion = math.sqrt(1.0 + self.beta_jump * j_val * h_ratio)
        effective_scale = max(0.1, min(10.0, scale * jump_cushion))
        return loc, effective_scale
