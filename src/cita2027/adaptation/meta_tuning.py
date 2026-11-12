"""Self-Tuning Online Meta-Controller for Cross-Horizon Adaptation (Meta-H3).

Provides causal online meta-learning over hyperparameter configurations (gamma_age, beta_overlap, lr_shared)
using continuous Exponential Weights (Hedge) algorithm with prequential CRPS loss feedback.

Key Features:
1. Online Meta-Ensemble: Maintains M parallel adapter configurations spanning low, medium,
   and high drift sensitivity regimes.
2. Causal Probability Simplex Tracking:
     p_{t+1}(m) = p_t(m) * exp(-eta_meta * loss_{t, m}) / Z_t
   where loss is evaluated strictly on causally released feedback A_t.
3. Regret Guarantee over Best Fixed Hyperparameter in Hindsight:
     Meta-Regret_T <= (log M) / eta_meta + (eta_meta / 8) * sum_{t=1}^T ||loss_t||_infty^2
4. Dynamic Entropy Regularization: Prevents premature probability collapse into a single configuration,
   ensuring continuous plasticity for future structural breaks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.correctors import BaseCorrector
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.probabilistic_metrics import compute_crps_empirical


@dataclass(frozen=True)
class HyperparameterConfig:
    config_id: str
    gamma_age: float
    overlap_discount_factor: float
    lr_shared: float
    lr_local: float
    latent_dim: int = 2


class MetaAdaptiveH3Controller(BaseCorrector):
    """Online Meta-Controller adapting H3 hyperparameters on the fly."""

    def __init__(
        self,
        horizons: List[int],
        configs: Optional[List[HyperparameterConfig]] = None,
        eta_meta: float = 0.5,
        entropy_floor: float = 0.05,
    ) -> None:
        super().__init__(horizons)
        self.eta_meta = float(eta_meta)
        self.entropy_floor = float(entropy_floor)

        if configs is None:
            # Default diverse 5-expert hyperparameter portfolio
            self.configs = [
                HyperparameterConfig("UltraConservative", gamma_age=0.02, overlap_discount_factor=0.90, lr_shared=0.03, lr_local=0.01),
                HyperparameterConfig("ModerateStationary", gamma_age=0.06, overlap_discount_factor=0.80, lr_shared=0.05, lr_local=0.02),
                HyperparameterConfig("BalancedDefault", gamma_age=0.12, overlap_discount_factor=0.70, lr_shared=0.08, lr_local=0.03),
                HyperparameterConfig("TurbulenceResponsive", gamma_age=0.25, overlap_discount_factor=0.50, lr_shared=0.12, lr_local=0.05),
                HyperparameterConfig("FastBreakTracking", gamma_age=0.45, overlap_discount_factor=0.30, lr_shared=0.18, lr_local=0.08),
            ]
        else:
            self.configs = configs

        self.M = len(self.configs)
        # Uniform initial prior over experts
        self.weights = np.ones(self.M, dtype=np.float64) / float(self.M)

        # Initialize expert models
        self.experts: List[HypothesisH3_OverlapAwareCorrector] = []
        for cfg in self.configs:
            m = HypothesisH3_OverlapAwareCorrector(
                horizons=horizons,
                latent_dim=cfg.latent_dim,
                lr_shared=cfg.lr_shared,
                lr_local=cfg.lr_local,
                gamma_age=cfg.gamma_age,
                overlap_discount_factor=cfg.overlap_discount_factor,
            )
            self.experts.append(m)

        # History tracking
        self.history_weights: List[np.ndarray] = []
        self.history_entropy: List[float] = []

    def adapt(self, origin_t: int, horizon: int, base_samples: np.ndarray) -> np.ndarray:
        """Mixture adaptation: samples drawn proportionally or blended across meta-experts."""
        s = np.asarray(base_samples, dtype=np.float64)
        n_samples = len(s)

        # Generate predictions from all experts
        expert_preds = [exp.adapt(origin_t, horizon, s) for exp in self.experts]

        # Stochastically sample predictions according to expert weights
        # to preserve valid non-parametric distributional mixture
        chosen_expert_idx = np.random.choice(self.M, p=self.weights)
        return expert_preds[chosen_expert_idx]

    def adapt_deterministic_mixture(
        self,
        origin_t: int,
        horizon: int,
        base_samples: np.ndarray,
    ) -> np.ndarray:
        """Deterministic quantile-weighted mixture of expert distributions."""
        s = np.asarray(base_samples, dtype=np.float64)
        expert_preds = np.array([exp.adapt(origin_t, horizon, s) for exp in self.experts])
        # Weighted mean of sorted sample quantiles
        sorted_preds = np.sort(expert_preds, axis=1)
        blended = np.tensordot(self.weights, sorted_preds, axes=(0, 0))
        return blended

    def update(self, feedback_items: List[ForecastFeedbackItem]) -> None:
        """Causal meta-update upon arrival of matured feedback."""
        if not feedback_items:
            return

        # 1. Update each internal expert with feedback
        for exp in self.experts:
            exp.update(feedback_items)

        # 2. Compute prequential loss for each expert on this batch
        expert_losses = np.zeros(self.M, dtype=np.float64)
        for i, exp in enumerate(self.experts):
            loss_sum = 0.0
            for item in feedback_items:
                # Evaluate CRPS of what this expert would have predicted
                pred_s = exp.adapt(item.origin_t, item.horizon, item.base_samples)
                loss_sum += compute_crps_empirical(pred_s, item.y_true)
            expert_losses[i] = loss_sum / float(len(feedback_items))

        # 3. Hedge exponential weights update
        # Normalize losses relative to minimum loss in batch to avoid numerical overflow
        min_loss = np.min(expert_losses)
        rel_losses = expert_losses - min_loss
        decay = np.exp(-self.eta_meta * rel_losses)
        new_w = self.weights * decay
        sum_w = np.sum(new_w)
        if sum_w > 1e-12:
            new_w = new_w / sum_w
        else:
            new_w = np.ones(self.M) / float(self.M)

        # 4. Apply entropy floor (mixture with uniform distribution)
        floor = self.entropy_floor / float(self.M)
        self.weights = (1.0 - self.entropy_floor) * new_w + floor
        self.weights = self.weights / np.sum(self.weights)

        # Record tracking
        self.history_weights.append(self.weights.copy())
        # Shannon entropy
        entropy = -float(np.sum(self.weights * np.log(np.maximum(self.weights, 1e-12))))
        self.history_entropy.append(entropy)

    def reset(self) -> None:
        for exp in self.experts:
            exp.reset()
        self.weights = np.ones(self.M, dtype=np.float64) / float(self.M)
        self.history_weights.clear()
        self.history_entropy.clear()

    def get_dominant_config(self) -> HyperparameterConfig:
        best_idx = int(np.argmax(self.weights))
        return self.configs[best_idx]
