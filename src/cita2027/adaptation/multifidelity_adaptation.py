"""
Asynchronous Multi-Fidelity Cross-Horizon Adaptation (MultiFidelity-H3).

Fuses zero-delay noisy proxy signals (low fidelity) with delayed high-precision
ground-truth verifications (high fidelity) into a unified cross-horizon representation manifold.

Theoretical Foundation (Theorem 11):
    Let forecast supervision arrive at two fidelity tiers:
    1. Low-Fidelity Stream (Proxy): Zero latency (tau=0), but high observation variance sigma_L^2;
    2. High-Fidelity Stream (Verified): Delayed by tau_H, but low observation variance sigma_H^2 << sigma_L^2.
    
    MultiFidelity-H3 employs optimal Gauss-Markov information fusion:
    Each incoming feedback event carries a precision weight:
        Omega_i = (1 / sigma_i^2) * exp( -gamma * age_i )
    
    When delayed high-fidelity verification arrives for an origin that previously
    received proxy feedback, MultiFidelity-H3 executes a Causal Innovation Correction:
        Delta z_t = eta * [ Omega_H * (y_H - hat{y}) - Omega_L * (y_L - hat{y}) ] * w_h
    
    Theorem 11 proves that multi-fidelity fusion strictly reduces mean squared tracking error:
        MSE(MultiFidelity) <= min( MSE(Proxy-Only), MSE(Verified-Only) )
    achieving the tracking speed of zero-latency proxies with the asymptotic accuracy
    of verified ground-truth.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector


class MultiFidelityFeedbackItem(ForecastFeedbackItem):
    """Feedback item carrying explicit fidelity tier metadata."""
    def __init__(
        self,
        origin_t: int,
        horizon: int,
        target_t: int,
        y_true: float,
        base_samples: np.ndarray,
        fidelity_tier: str = "high",  # "low" or "high"
        noise_variance: float = 0.1,
    ):
        super().__init__(
            origin_t=origin_t,
            horizon=horizon,
            target_t=target_t,
            y_true=y_true,
            base_samples=base_samples,
        )
        self.fidelity_tier = fidelity_tier
        self.noise_variance = noise_variance


class MultiFidelityCrossHorizonAdapter:
    """
    Cross-horizon adapter with asynchronous multi-fidelity information fusion (MultiFidelity-H3).
    """

    def __init__(
        self,
        horizons: List[int],
        latent_dim: int = 2,
        lr_shared: float = 0.08,
        gamma_age: float = 0.10,
        low_fidelity_variance: float = 1.0,
        high_fidelity_variance: float = 0.05,
    ):
        self.horizons = sorted(list(horizons))
        self.latent_dim = latent_dim
        self.lr_shared = lr_shared
        self.gamma_age = gamma_age
        self.var_low = low_fidelity_variance
        self.var_high = high_fidelity_variance
        self.prior_precision = 1.0

        self.base_corrector = HypothesisH3_OverlapAwareCorrector(
            horizons=horizons,
            latent_dim=latent_dim,
            lr_shared=lr_shared,
            gamma_age=gamma_age,
        )

        # Cache of proxy feedback to compute innovation corrections upon high-fidelity arrival
        self.proxy_cache: Dict[Tuple[int, int], float] = {}

    def update_with_feedback(
        self,
        feedback_items: List[ForecastFeedbackItem],
        current_t: int,
    ) -> Dict[str, float]:
        """
        Process incoming low-fidelity and high-fidelity feedback items.
        """
        if not feedback_items:
            return {"fusion_weight_ratio": 1.0, "state_norm": float(np.linalg.norm(self.base_corrector.shared_state))}

        total_high = 0
        total_low = 0
        fused_gradient = np.zeros(self.latent_dim, dtype=np.float64)

        for item in feedback_items:
            h = item.horizon
            w_h = self.base_corrector.w[h]
            loc = getattr(item, "base_location", None)
            if loc is None:
                loc = float(np.median(item.base_samples))

            pred_loc = loc + float(np.dot(w_h, self.base_corrector.shared_state))
            residual = float(item.y_true - pred_loc)

            tier = getattr(item, "fidelity_tier", "high")
            age = max(0, current_t - item.origin_t)
            age_decay = math.exp(-self.gamma_age * age)

            if tier == "low":
                total_low += 1
                prec = (1.0 / self.var_low) * age_decay
                fused_gradient += prec * residual * w_h
                # Cache proxy value
                self.proxy_cache[(item.origin_t, h)] = item.y_true
            else:
                total_high += 1
                prec = (1.0 / self.var_high) * age_decay
                # Check if we previously had a proxy for this forecast
                prev_proxy = self.proxy_cache.pop((item.origin_t, h), None)
                if prev_proxy is not None:
                    # Subtract proxy contribution, add verified ground truth
                    proxy_res = prev_proxy - pred_loc
                    proxy_prec = (1.0 / self.var_low) * age_decay
                    fused_gradient += (prec * residual - proxy_prec * proxy_res) * w_h
                else:
                    fused_gradient += prec * residual * w_h

        # Gauss-Markov precision weighting including prior precision
        total_prec = self.prior_precision + (total_low / self.var_low + total_high / self.var_high)
        normalized_step = fused_gradient / total_prec

        # Overlap scaling
        k_eff = self.base_corrector.compute_effective_overlap_factor(feedback_items)
        lr_eff = self.lr_shared / math.sqrt(max(1.0, k_eff))

        # Update state
        self.base_corrector.shared_state += lr_eff * normalized_step

        # Enforce compactness ||z|| <= 8.0
        norm_z = float(np.linalg.norm(self.base_corrector.shared_state))
        if norm_z > 8.0:
            self.base_corrector.shared_state = (self.base_corrector.shared_state / norm_z) * 8.0

        return {
            "low_fidelity_count": float(total_low),
            "high_fidelity_count": float(total_high),
            "effective_gradient_norm": float(np.linalg.norm(fused_gradient)),
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
