"""Fractional Brownian Motion & Long-Memory Hurst Exponent Adaptation (Hurst-H3).

Implements causal online Hurst exponent estimation and power-law memory
decay kernel adaptation for time series exhibiting long-range dependence
(H in (0.5, 1.0), e.g., hydrological flows, network telemetries, volatility memory)
or anti-persistent mean-reversion (H in (0, 0.5)).

Theorem 21 (Fractional Memory Regret & Convergence):
Under fractional Gaussian innovations B_H(t) with Hurst parameter H in (0, 1),
the Hurst-adapted online cross-horizon update with power-law weights
omega_t(h) = h^{2H - 2} achieves dynamic regret:
    R_T(Hurst-H3) <= O( T^{max(1/2, H)} * K_eff_bar^{1 - H} + (D_Z / eta_T) * P_T ),
eliminating the asymptotic long-memory bias accumulation that degrades
standard exponential-decay (Markovian) online learners.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


class OnlineHurstEstimator:
    """Estimates the Hurst exponent H in (0, 1) online using multi-horizon variance scaling.

    In fractional processes, Var(y_{t+h} - y_t) ~ c * h^{2H}.
    Taking logs: log Var = 2H * log(h) + const.
    """

    def __init__(self, horizons: Sequence[int] = (1, 2, 4, 8, 16), window_size: int = 40):
        self.horizons = sorted(list(horizons))
        self.window_size = int(window_size)
        # Store recent squared differences per horizon
        self.sq_diffs: Dict[int, List[float]] = {h: [] for h in self.horizons}
        self.current_hurst = 0.5  # Standard Brownian motion prior
        self.log_h = np.log(np.array(self.horizons, dtype=np.float64))

    def observe(self, horizon: int, squared_error: float):
        if horizon in self.sq_diffs:
            self.sq_diffs[horizon].append(float(squared_error))
            if len(self.sq_diffs[horizon]) > self.window_size:
                self.sq_diffs[horizon].pop(0)

        # Update Hurst estimation if sufficient history exists across at least 3 horizons
        valid_h = [h for h in self.horizons if len(self.sq_diffs[h]) >= 5]
        if len(valid_h) >= 3:
            log_vars = []
            log_h_valid = []
            for h in valid_h:
                mean_v = float(np.mean(self.sq_diffs[h]))
                log_vars.append(math.log(max(mean_v, 1e-8)))
                log_h_valid.append(math.log(h))

            # Linear regression: log_vars = 2H * log_h + intercept
            x = np.array(log_h_valid)
            y = np.array(log_vars)
            n = len(x)
            x_mean = float(np.mean(x))
            y_mean = float(np.mean(y))
            slope = float(np.sum((x - x_mean) * (y - y_mean)) / max(np.sum((x - x_mean)**2), 1e-8))

            # H = slope / 2.0, clipped to [0.05, 0.95]
            raw_h = slope / 2.0
            self.current_hurst = float(np.clip(raw_h, 0.05, 0.95))

        return self.current_hurst


class FractionalBrownianCrossHorizonAdapter:
    """Hypothesis H3 implementation with Fractional Long-Memory Hurst Adaptation."""

    def __init__(
        self,
        horizons: Sequence[int] = (1, 2, 4, 8, 16),
        latent_dim: int = 4,
        learning_rate: float = 0.04,
        state_bound: float = 8.0,
        min_scale: float = 0.1,
        max_scale: float = 10.0,
    ):
        self.horizons = list(horizons)
        self.latent_dim = int(latent_dim)
        self.learning_rate = float(learning_rate)
        self.state_bound = float(state_bound)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

        self.hurst_estimator = OnlineHurstEstimator(horizons=self.horizons)

        # State vector
        self.z = np.zeros(self.latent_dim, dtype=np.float64)

        # Projection vectors w_h
        self.w: Dict[int, np.ndarray] = {}
        for idx, h in enumerate(self.horizons):
            vec = np.zeros(self.latent_dim, dtype=np.float64)
            vec[idx % self.latent_dim] = 1.0
            self.w[h] = vec / (np.linalg.norm(vec) + 1e-8)

        # Scales
        self.log_scales: Dict[int, float] = {h: 0.0 for h in self.horizons}
        self.total_updates = 0

    def compute_fractional_weight(self, horizon: int, hurst: float) -> float:
        """Compute the power-law memory weight: omega(h) = h^{2H - 2}."""
        power = 2.0 * hurst - 2.0
        # h^power
        return float(horizon ** power)

    def update_feedback(self, feedback_items: List[Dict[str, float]]) -> Dict[str, float]:
        """Update state using fractional memory kernel weighting."""
        if not feedback_items:
            return {
                "num_updates": 0,
                "estimated_hurst": self.hurst_estimator.current_hurst,
                "z_norm": float(np.linalg.norm(self.z)),
            }

        # Update Hurst estimator first
        for item in feedback_items:
            h = int(item["horizon"])
            res = float(item["y_true"]) - float(item["y_pred_loc"])
            self.hurst_estimator.observe(h, res**2)

        current_h = self.hurst_estimator.current_hurst

        weighted_grads = []
        weights = []

        for item in feedback_items:
            h = int(item["horizon"])
            y = float(item["y_true"])
            y_loc = float(item["y_pred_loc"])
            disp = max(float(item.get("y_pred_disp", 1.0)), 1e-6)
            res = y - y_loc

            proj_w = self.w.get(h, np.ones(self.latent_dim) / math.sqrt(self.latent_dim))
            pred = float(np.dot(proj_w, self.z))
            err = pred - res

            # Fractional memory weight
            omega = self.compute_fractional_weight(h, current_h)
            weights.append(omega)
            weighted_grads.append(omega * err * proj_w)

            # Scale update
            z_err = abs(res) / disp
            target_z = math.sqrt(2.0 / math.pi)
            scale_grad = np.clip(z_err - target_z, -2.0, 2.0)
            self.log_scales[h] = float(np.clip(
                self.log_scales[h] + 0.02 * scale_grad,
                math.log(self.min_scale),
                math.log(self.max_scale),
            ))

        # Effective Kish sample size under power-law weights
        sum_w = sum(weights)
        sum_w_sq = sum(w**2 for w in weights)
        k_eff = (sum_w**2) / max(sum_w_sq, 1e-8)

        # Scaled gradient step
        mean_grad = np.sum(weighted_grads, axis=0) / max(sum_w, 1e-8)
        # Fractional step size adjustment: eta / K_eff^(1 - H)
        adaptive_lr = self.learning_rate / (max(k_eff, 1.0) ** (1.0 - current_h))

        self.z = self.z - adaptive_lr * mean_grad

        # Lyapunov projection
        z_norm = np.linalg.norm(self.z)
        if z_norm > self.state_bound:
            self.z = self.z * (self.state_bound / (z_norm + 1e-12))

        self.total_updates += len(feedback_items)

        return {
            "num_updates": len(feedback_items),
            "estimated_hurst": current_h,
            "effective_k": float(k_eff),
            "z_norm": float(np.linalg.norm(self.z)),
        }

    def adapt_forecast(
        self,
        horizon: int,
        base_samples: np.ndarray,
        base_location: float,
        base_dispersion: float,
    ) -> Tuple[np.ndarray, float, float]:
        """Apply fractional adapted shift and dispersion to base samples."""
        proj_w = self.w.get(horizon, np.ones(self.latent_dim) / math.sqrt(self.latent_dim))
        bias = float(np.dot(proj_w, self.z))

        scale_mult = float(np.exp(self.log_scales.get(horizon, 0.0)))
        scale_mult = float(np.clip(scale_mult, self.min_scale, self.max_scale))

        adapted_samples = base_location + bias + scale_mult * (base_samples - base_location)
        adapted_loc = float(np.mean(adapted_samples))
        adapted_disp = float(np.std(adapted_samples))

        return adapted_samples, adapted_loc, adapted_disp
