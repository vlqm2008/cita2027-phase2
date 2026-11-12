"""Adaptation correctors: Baselines B0, B1, B2, B3 and Hypothesis H1 Shared-State model."""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


class BaseCorrector:
    """Abstract base class for multi-horizon forecast adaptation."""

    def __init__(self, horizons: list[int]):
        self.horizons = sorted(horizons)

    def adapt(self, origin_t: int, horizon: int, base_samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, feedback_items: list) -> None:
        """Update internal state using mature feedback items available at current timestamp."""
        raise NotImplementedError

    def reset(self) -> None:
        pass


class BaselineB0_Frozen(BaseCorrector):
    """B0: Frozen foundation model, no adaptation."""

    def adapt(self, origin_t: int, horizon: int, base_samples: np.ndarray) -> np.ndarray:
        return np.asarray(base_samples, dtype=np.float64).copy()

    def update(self, feedback_items: list) -> None:
        pass


class BaselineB1_IndependentEWMA(BaseCorrector):
    """B1: Horizon-specific independent EWMA bias correction."""

    def __init__(self, horizons: list[int], alpha: float = 0.10):
        super().__init__(horizons)
        self.alpha = alpha
        self.bias: dict[int, float] = {h: 0.0 for h in self.horizons}

    def adapt(self, origin_t: int, horizon: int, base_samples: np.ndarray) -> np.ndarray:
        s = np.asarray(base_samples, dtype=np.float64)
        return s + self.bias[horizon]

    def update(self, feedback_items: list) -> None:
        for item in feedback_items:
            h = item.horizon
            pred_med = float(np.median(item.base_samples))
            err = item.y_true - pred_med
            self.bias[h] = (1.0 - self.alpha) * self.bias[h] + self.alpha * err

    def reset(self) -> None:
        self.bias = {h: 0.0 for h in self.horizons}


class BaselineB2_IndependentLocScale(BaseCorrector):
    """B2: Horizon-specific independent location and positive scale correction."""

    def __init__(self, horizons: list[int], alpha: float = 0.10):
        super().__init__(horizons)
        self.alpha = alpha
        self.bias: dict[int, float] = {h: 0.0 for h in self.horizons}
        self.scale_mult: dict[int, float] = {h: 1.0 for h in self.horizons}

    def adapt(self, origin_t: int, horizon: int, base_samples: np.ndarray) -> np.ndarray:
        s = np.asarray(base_samples, dtype=np.float64)
        loc = float(np.median(s))
        scale = max(0.2, min(5.0, self.scale_mult[horizon]))
        return loc + self.bias[horizon] + scale * (s - loc)

    def update(self, feedback_items: list) -> None:
        for item in feedback_items:
            h = item.horizon
            s = np.asarray(item.base_samples, dtype=np.float64)
            loc = float(np.median(s))
            base_disp = float(np.std(s)) if len(s) > 1 else 1.0
            base_disp = max(1e-4, base_disp)

            err = item.y_true - loc
            self.bias[h] = (1.0 - self.alpha) * self.bias[h] + self.alpha * err

            obs_disp = abs(item.y_true - loc) * np.sqrt(np.pi / 2.0)
            target_scale = obs_disp / base_disp
            self.scale_mult[h] = (1.0 - self.alpha) * self.scale_mult[h] + self.alpha * target_scale

    def reset(self) -> None:
        self.bias = {h: 0.0 for h in self.horizons}
        self.scale_mult = {h: 1.0 for h in self.horizons}


class BaselineB3_FullyPooled(BaseCorrector):
    """B3: Fully pooled scalar correction across all horizons."""

    def __init__(self, horizons: list[int], alpha: float = 0.05):
        super().__init__(horizons)
        self.alpha = alpha
        self.pooled_bias = 0.0
        self.pooled_scale = 1.0

    def adapt(self, origin_t: int, horizon: int, base_samples: np.ndarray) -> np.ndarray:
        s = np.asarray(base_samples, dtype=np.float64)
        loc = float(np.median(s))
        scale = max(0.2, min(5.0, self.pooled_scale))
        return loc + self.pooled_bias + scale * (s - loc)

    def update(self, feedback_items: list) -> None:
        for item in feedback_items:
            s = np.asarray(item.base_samples, dtype=np.float64)
            loc = float(np.median(s))
            err = item.y_true - loc
            self.pooled_bias = (1.0 - self.alpha) * self.pooled_bias + self.alpha * err

            base_disp = max(1e-4, float(np.std(s)))
            target_scale = (abs(err) * np.sqrt(np.pi / 2.0)) / base_disp
            self.pooled_scale = (1.0 - self.alpha) * self.pooled_scale + self.alpha * target_scale

    def reset(self) -> None:
        self.pooled_bias = 0.0
        self.pooled_scale = 1.0


class HypothesisH1_SharedStateCorrector(BaseCorrector):
    """Hypothesis H1: Cross-Horizon Transfer via Low-Rank Shared State Decomposition.

    Mechanism:
      bias[t, h] = w_h^T s_t + b_h^{local}
      scale[t, h] = exp(u_h^T g_t + c_h^{local})  (strictly positive)

    When asynchronous feedback from ANY horizon h arrives, shared state s_t updates,
    instantly propagating structural shift information to all pending horizons.
    """

    def __init__(
        self,
        horizons: list[int],
        latent_dim: int = 2,
        lr_shared: float = 0.08,
        lr_local: float = 0.03,
    ):
        super().__init__(horizons)
        self.latent_dim = latent_dim
        self.lr_shared = lr_shared
        self.lr_local = lr_local

        # Low-dimensional horizon basis w_h of dimension latent_dim
        self.w: dict[int, np.ndarray] = {}
        for h in self.horizons:
            v = np.array([(np.log2(max(1, h)) / 4.0) ** k for k in range(latent_dim)], dtype=np.float64)
            norm = np.linalg.norm(v)
            self.w[h] = v / norm if norm > 1e-8 else v

        # Shared state vector s_t
        self.shared_state = np.zeros(latent_dim, dtype=np.float64)
        self.local_bias = {h: 0.0 for h in self.horizons}

        # Shared log-scale state g_t
        self.shared_scale_state = np.zeros(latent_dim, dtype=np.float64)
        self.local_log_scale = {h: 0.0 for h in self.horizons}

    def adapt(self, origin_t: int, horizon: int, base_samples: np.ndarray) -> np.ndarray:
        s = np.asarray(base_samples, dtype=np.float64)
        loc = float(np.median(s))

        # bias = shared + local
        bias = float(np.dot(self.w[horizon], self.shared_state) + self.local_bias[horizon])

        # scale multiplier = exp(shared + local)
        log_scale = float(np.dot(self.w[horizon], self.shared_scale_state) + self.local_log_scale[horizon])
        scale_mult = float(np.exp(np.clip(log_scale, -1.6, 1.6)))  # strictly positive in [0.2, 5.0]

        return loc + bias + scale_mult * (s - loc)

    def update(self, feedback_items: list) -> None:
        for item in feedback_items:
            h = item.horizon
            s = np.asarray(item.base_samples, dtype=np.float64)
            loc = float(np.median(s))
            w_h = self.w[h]

            # Current prediction error under adapted state
            current_bias = float(np.dot(w_h, self.shared_state) + self.local_bias[h])
            residual = item.y_true - (loc + current_bias)

            # Update shared state (cross-horizon transfer step)
            self.shared_state += self.lr_shared * w_h * residual

            # Update local residual bias
            self.local_bias[h] += self.lr_local * residual

            # Update scale states
            base_disp = max(1e-4, float(np.std(s)))
            current_log_scale = float(np.dot(w_h, self.shared_scale_state) + self.local_log_scale[h])
            current_scale = np.exp(current_log_scale)
            obs_disp = abs(item.y_true - loc) * np.sqrt(np.pi / 2.0)
            target_scale = max(0.2, min(5.0, obs_disp / base_disp))
            log_scale_err = np.log(target_scale) - current_log_scale

            self.shared_scale_state += self.lr_shared * w_h * log_scale_err
            self.local_log_scale[h] += self.lr_local * log_scale_err

    def reset(self) -> None:
        self.shared_state.fill(0.0)
        self.shared_scale_state.fill(0.0)
        self.local_bias = {h: 0.0 for h in self.horizons}
        self.local_log_scale = {h: 0.0 for h in self.horizons}
