"""Hypothesis H2: Age-decay and drift-consistency weighting for asynchronous feedback."""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from cita2027.adaptation.correctors import BaseCorrector
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


class HypothesisH2_AgeWeightedCorrector(BaseCorrector):
    """Hypothesis H2: Mitigation of Stale-Feedback Negative Transfer.

    Weighting mechanisms for feedback arriving with delay (age = target_t - origin_t = h):
      - 'exponential': w = exp(-gamma * age)
      - 'hyperbolic': w = 1.0 / (1.0 + gamma * age)
      - 'drift_aware': w = age_weight * consistency_factor
        where consistency_factor dampens feedback if residual sign opposes recent short-horizon trend.
    """

    def __init__(
        self,
        horizons: list[int],
        latent_dim: int = 2,
        lr_shared: float = 0.08,
        lr_local: float = 0.03,
        decay_type: str = "exponential",
        gamma: float = 0.15,
        use_drift_consistency: bool = False,
    ):
        super().__init__(horizons)
        self.latent_dim = latent_dim
        self.lr_shared = lr_shared
        self.lr_local = lr_local
        self.decay_type = decay_type
        self.gamma = gamma
        self.use_drift_consistency = use_drift_consistency

        # Normalized horizon embeddings
        self.w: dict[int, np.ndarray] = {}
        for h in self.horizons:
            v = np.array([1.0, np.log2(max(1, h)) / 4.0], dtype=np.float64)
            if latent_dim == 1:
                v = v[:1]
            self.w[h] = v / np.linalg.norm(v)

        self.shared_state = np.zeros(latent_dim, dtype=np.float64)
        self.local_bias = {h: 0.0 for h in self.horizons}
        self.shared_scale_state = np.zeros(latent_dim, dtype=np.float64)
        self.local_log_scale = {h: 0.0 for h in self.horizons}

        # Track recent innovation direction from fastest feedback (h=1) to measure consistency
        self.recent_short_horizon_trend = 0.0

    def compute_weight(self, item: ForecastFeedbackItem, residual: float) -> float:
        age = float(max(1, item.target_t - item.origin_t))

        if self.decay_type == "uniform":
            w_age = 1.0
        elif self.decay_type == "exponential":
            w_age = float(np.exp(-self.gamma * (age - 1.0)))
        elif self.decay_type == "hyperbolic":
            w_age = float(1.0 / (1.0 + self.gamma * (age - 1.0)))
        else:
            w_age = 1.0

        # Drift consistency check: if feedback residual strongly opposes recent trend,
        # it is likely stale feedback from a pre-reversal regime -> downweight aggressively
        w_consistency = 1.0
        if self.use_drift_consistency and abs(self.recent_short_horizon_trend) > 1e-4:
            alignment = np.sign(residual) * np.sign(self.recent_short_horizon_trend)
            if alignment < 0:
                # Disagreement with current short-horizon regime
                w_consistency = 0.25

        return float(np.clip(w_age * w_consistency, 0.05, 1.0))

    def adapt(self, origin_t: int, horizon: int, base_samples: np.ndarray) -> np.ndarray:
        s = np.asarray(base_samples, dtype=np.float64)
        loc = float(np.median(s))

        bias = float(np.dot(self.w[horizon], self.shared_state) + self.local_bias[horizon])
        log_scale = float(np.dot(self.w[horizon], self.shared_scale_state) + self.local_log_scale[horizon])
        scale_mult = float(np.exp(np.clip(log_scale, -1.6, 1.6)))

        return loc + bias + scale_mult * (s - loc)

    def update(self, feedback_items: list[ForecastFeedbackItem]) -> None:
        for item in feedback_items:
            h = item.horizon
            s = np.asarray(item.base_samples, dtype=np.float64)
            loc = float(np.median(s))
            w_h = self.w[h]

            current_bias = float(np.dot(w_h, self.shared_state) + self.local_bias[h])
            residual = item.y_true - (loc + current_bias)

            # Update short-horizon trend memory
            if h == min(self.horizons):
                self.recent_short_horizon_trend = 0.8 * self.recent_short_horizon_trend + 0.2 * residual

            # Compute age/drift-aware weight
            weight = self.compute_weight(item, residual)

            # Age-dampened update steps
            self.shared_state += (self.lr_shared * weight) * w_h * residual
            self.local_bias[h] += (self.lr_local * weight) * residual

            # Scale adaptation
            base_disp = max(1e-4, float(np.std(s)))
            current_log_scale = float(np.dot(w_h, self.shared_scale_state) + self.local_log_scale[h])
            obs_disp = abs(item.y_true - loc) * np.sqrt(np.pi / 2.0)
            target_scale = max(0.2, min(5.0, obs_disp / base_disp))
            log_scale_err = np.log(target_scale) - current_log_scale

            self.shared_scale_state += (self.lr_shared * weight) * w_h * log_scale_err
            self.local_log_scale[h] += (self.lr_local * weight) * log_scale_err

    def reset(self) -> None:
        self.shared_state.fill(0.0)
        self.shared_scale_state.fill(0.0)
        self.local_bias = {h: 0.0 for h in self.horizons}
        self.local_log_scale = {h: 0.0 for h in self.horizons}
        self.recent_short_horizon_trend = 0.0
