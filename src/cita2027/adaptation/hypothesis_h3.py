"""Hypothesis H3: Overlap-aware update normalization for multi-horizon feedback."""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from cita2027.adaptation.correctors import BaseCorrector
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


class HypothesisH3_OverlapAwareCorrector(BaseCorrector):
    """Hypothesis H3: Overlap-Aware Update Stability.

    When multiple forecasts mature at the same timestamp t (or from overlapping intervals),
    their errors are highly dependent.
    Instead of summing unscaled updates (which blows up variance), H3 computes the effective
    information degree K_eff and scales learning rate by 1 / sqrt(K_eff).
    """

    def __init__(
        self,
        horizons: list[int],
        latent_dim: int = 2,
        lr_shared: float = 0.08,
        lr_local: float = 0.03,
        gamma_age: float = 0.12,
        overlap_discount_factor: float = 0.70,
    ):
        super().__init__(horizons)
        self.latent_dim = latent_dim
        self.lr_shared = lr_shared
        self.lr_local = lr_local
        self.gamma_age = gamma_age
        self.overlap_discount_factor = overlap_discount_factor

        self.w: dict[int, np.ndarray] = {}
        for h in self.horizons:
            v = np.array([(np.log2(max(1, h)) / 4.0) ** k for k in range(latent_dim)], dtype=np.float64)
            self.w[h] = v / np.linalg.norm(v)

        self.shared_state = np.zeros(latent_dim, dtype=np.float64)
        self.local_bias = {h: 0.0 for h in self.horizons}
        self.shared_scale_state = np.zeros(latent_dim, dtype=np.float64)
        self.local_log_scale = {h: 0.0 for h in self.horizons}

        # Stability tracking metrics
        self.state_step_history: list[float] = []

    def compute_effective_overlap_factor(self, items: list[ForecastFeedbackItem]) -> float:
        """Calculate effective sample size K_eff accounting for horizon overlap."""
        k = len(items)
        if k <= 1:
            return 1.0

        # Correlation proxy between horizons: closer horizons share more information
        total_overlap = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                h_i = items[i].horizon
                h_j = items[j].horizon
                # Overlap proxy based on horizon ratio
                ratio = min(h_i, h_j) / max(h_i, h_j)
                total_overlap += 2.0 * ratio * self.overlap_discount_factor

        k_eff = float(k + total_overlap)
        return max(1.0, k_eff)

    def adapt(self, origin_t: int, horizon: int, base_samples: np.ndarray) -> np.ndarray:
        s = np.asarray(base_samples, dtype=np.float64)
        loc = float(np.median(s))

        bias = float(np.dot(self.w[horizon], self.shared_state) + self.local_bias[horizon])
        log_scale = float(np.dot(self.w[horizon], self.shared_scale_state) + self.local_log_scale[horizon])
        scale_mult = float(np.exp(np.clip(log_scale, -1.6, 1.6)))

        return loc + bias + scale_mult * (s - loc)

    def update(self, feedback_items: list[ForecastFeedbackItem]) -> None:
        if not feedback_items:
            return

        # 1. Overlap normalization factor
        k_eff = self.compute_effective_overlap_factor(feedback_items)
        overlap_scale = 1.0 / np.sqrt(k_eff)

        # 2. Accumulate gradient step across feedback items in batch
        delta_shared = np.zeros(self.latent_dim, dtype=np.float64)
        delta_shared_scale = np.zeros(self.latent_dim, dtype=np.float64)

        for item in feedback_items:
            h = item.horizon
            s = np.asarray(item.base_samples, dtype=np.float64)
            loc = float(np.median(s))
            w_h = self.w[h]

            current_bias = float(np.dot(w_h, self.shared_state) + self.local_bias[h])
            residual = item.y_true - (loc + current_bias)

            # Age decay weight (H2)
            age = float(max(1, item.target_t - item.origin_t))
            w_age = float(np.exp(-self.gamma_age * (age - 1.0)))

            # Accumulate scaled updates
            weight = w_age * overlap_scale
            delta_shared += (self.lr_shared * weight) * w_h * residual
            self.local_bias[h] += (self.lr_local * weight) * residual

            # Scale adaptation
            base_disp = max(1e-4, float(np.std(s)))
            current_log_scale = float(np.dot(w_h, self.shared_scale_state) + self.local_log_scale[h])
            obs_disp = abs(item.y_true - loc) * np.sqrt(np.pi / 2.0)
            target_scale = max(0.2, min(5.0, obs_disp / base_disp))
            log_scale_err = np.log(target_scale) - current_log_scale

            delta_shared_scale += (self.lr_shared * weight) * w_h * log_scale_err
            self.local_log_scale[h] += (self.lr_local * weight) * log_scale_err

        # Apply bounded step
        self.shared_state += delta_shared
        self.shared_scale_state += delta_shared_scale

        step_norm = float(np.linalg.norm(delta_shared))
        self.state_step_history.append(step_norm)

    def reset(self) -> None:
        self.shared_state.fill(0.0)
        self.shared_scale_state.fill(0.0)
        self.local_bias = {h: 0.0 for h in self.horizons}
        self.local_log_scale = {h: 0.0 for h in self.horizons}
        self.state_step_history.clear()
