"""
Cross-Horizon Shared-State Probabilistic Adapters (Baselines B3, B4 & Hypotheses H1, H2, H3).

Implements:
- Baseline B3: Fully pooled correction shared across horizons
- Baseline B4: Partially shared static correction model (without asynchronous weighting)
- Hypothesis H1: Cross-horizon manifold transfer
- Hypothesis H2: Age-decay and drift-consistency gating
- Hypothesis H3: Unified Overlap-aware Kish normalized adapter
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from cita2027.adaptation.correctors import (
    BaseCorrector,
    BaselineB3_FullyPooled,
    HypothesisH1_SharedStateCorrector,
)
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h2 import HypothesisH2_AgeWeightedCorrector as HypothesisH2_AgeDecayCorrector
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector


class BaselineB4_PartiallySharedStatic(BaseCorrector):
    """
    Baseline B4: Partially shared correction model without asynchronous age/overlap mechanism.
    
    Maintains a low-rank shared state across horizons, but applies uniform weights
    to all arriving feedback without age discounting or overlap normalization.
    """

    def __init__(
        self,
        horizons: List[int],
        latent_dim: int = 2,
        lr_shared: float = 0.08,
        lr_local: float = 0.03,
    ):
        super().__init__(horizons)
        self.latent_dim = latent_dim
        self.lr_shared = lr_shared
        self.lr_local = lr_local

        self.w: Dict[int, np.ndarray] = {}
        for h in self.horizons:
            v = np.array([(np.log2(max(1, h)) / 4.0) ** k for k in range(latent_dim)], dtype=np.float64)
            self.w[h] = v / np.linalg.norm(v)

        self.shared_state = np.zeros(latent_dim, dtype=np.float64)
        self.local_bias = {h: 0.0 for h in self.horizons}
        self.shared_scale_state = np.zeros(latent_dim, dtype=np.float64)
        self.local_log_scale = {h: 0.0 for h in self.horizons}

    def reset(self) -> None:
        self.shared_state = np.zeros(self.latent_dim, dtype=np.float64)
        self.local_bias = {h: 0.0 for h in self.horizons}
        self.shared_scale_state = np.zeros(self.latent_dim, dtype=np.float64)
        self.local_log_scale = {h: 0.0 for h in self.horizons}

    def adapt(self, origin_t: int, horizon: int, base_samples: np.ndarray) -> np.ndarray:
        s = np.asarray(base_samples, dtype=np.float64)
        loc = float(np.median(s))
        w_h = self.w[horizon]
        bias = float(np.dot(w_h, self.shared_state) + self.local_bias[horizon])
        log_scale = float(np.dot(w_h, self.shared_scale_state) + self.local_log_scale[horizon])
        scale_mult = float(np.clip(np.exp(log_scale), 0.1, 10.0))
        return loc + bias + scale_mult * (s - loc)

    def update(self, feedback_items: List[ForecastFeedbackItem]) -> None:
        if not feedback_items:
            return
        # Static uniform update without age or overlap weighting
        delta_shared = np.zeros(self.latent_dim, dtype=np.float64)
        delta_shared_scale = np.zeros(self.latent_dim, dtype=np.float64)

        for item in feedback_items:
            h = item.horizon
            s = np.asarray(item.base_samples, dtype=np.float64)
            loc = float(np.median(s))
            w_h = self.w[h]
            current_bias = float(np.dot(w_h, self.shared_state) + self.local_bias[h])
            residual = item.y_true - (loc + current_bias)

            delta_shared += self.lr_shared * w_h * residual
            self.local_bias[h] += self.lr_local * residual

            base_disp = max(1e-4, float(np.std(s)))
            current_log_scale = float(np.dot(w_h, self.shared_scale_state) + self.local_log_scale[h])
            obs_disp = abs(item.y_true - loc) * np.sqrt(np.pi / 2.0)
            target_scale = max(0.2, min(5.0, obs_disp / base_disp))
            log_scale_err = np.log(target_scale) - current_log_scale

            delta_shared_scale += self.lr_shared * w_h * log_scale_err
            self.local_log_scale[h] += self.lr_local * log_scale_err

        self.shared_state += delta_shared
        self.shared_scale_state += delta_shared_scale
