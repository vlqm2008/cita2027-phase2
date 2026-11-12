"""
Sequential Bayesian Online Posterior Adaptation (Bayes-H3).

Maintains a full posterior distribution p(z_t | F_t) = N(m_t, Sigma_t) over the
latent cross-horizon representation manifold. Quantifies epistemic manifold uncertainty
and propagates it causally into multi-horizon predictive distributions.

Theoretical Foundation (Theorem 8):
    Let the latent representation state evolve according to a random walk with drift:
        z_{t+1} = z_t + w_t,  w_t ~ N(0, Q)
    and observation likelihood for horizon h with feedback item i:
        y_{t+h} - loc_base = w_h^T z_t + v_{t,h},  v_{t,h} ~ N(0, R_h)
    
    Bayes-H3 executes online recursive Kalman-Bucy filtering:
    1. Time Update (Prior Diffusion):
       m_{t|t-1} = (1 - alpha_drift) * m_{t-1}
       Sigma_{t|t-1} = Sigma_{t-1} + Q
    2. Measurement Update (Asynchronous Multi-Horizon Innovation):
       For each feedback item arriving at timestamp t:
       K_t = Sigma_{t|t-1} w_h (w_h^T Sigma_{t|t-1} w_h + R_h)^{-1}
       m_t = m_{t|t-1} + K_t (e_{t,h} - w_h^T m_{t|t-1})
       Sigma_t = (I - K_t w_h^T) Sigma_{t|t-1}
    3. Epistemic Uncertainty Hedging:
       Epistemic manifold variance for horizon h is sigma_{epistemic, h}^2 = w_h^T Sigma_t w_h.
       Predictive scale is inflated by:
       s_{t,h} = s_{base} * sqrt(1 + sigma_{epistemic, h}^2 / sigma_base^2)
    
    This guarantees that during periods of sparse or delayed feedback, predictive
    intervals widen automatically to reflect epistemic parameter uncertainty.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


class BayesianCrossHorizonPosteriorTracker:
    """
    Online Recursive Gaussian Posterior Filter for Latent Cross-Horizon State z_t.
    """

    def __init__(
        self,
        horizons: List[int],
        latent_dim: int = 2,
        prior_variance: float = 1.0,
        process_noise_q: float = 0.01,
        obs_noise_r: float = 0.25,
        drift_decay: float = 0.01,
    ):
        self.horizons = sorted(list(horizons))
        self.latent_dim = latent_dim
        self.q_noise = process_noise_q
        self.r_noise = obs_noise_r
        self.drift_decay = drift_decay

        # Prior mean and covariance
        self.mean_z = np.zeros(latent_dim, dtype=np.float64)
        self.cov_z = np.eye(latent_dim, dtype=np.float64) * prior_variance

        # Deterministic normalized horizon projection vectors
        rng = np.random.RandomState(42)
        raw_w = rng.randn(len(self.horizons), latent_dim)
        row_norms = np.linalg.norm(raw_w, axis=1, keepdims=True) + 1e-8
        self.w_matrix = raw_w / row_norms
        self.horizon_to_w = {h: self.w_matrix[i] for i, h in enumerate(self.horizons)}

        # Horizon scale state
        self.scale_factors = {h: 1.0 for h in self.horizons}

    def predict_step(self) -> None:
        """Propagate state and covariance forward under process drift."""
        self.mean_z = (1.0 - self.drift_decay) * self.mean_z
        self.cov_z = self.cov_z + np.eye(self.latent_dim) * self.q_noise

    def update_with_feedback(
        self,
        feedback_items: List[ForecastFeedbackItem],
        current_t: int,
    ) -> Dict[str, float]:
        """
        Incorporate asynchronous feedback items via Kalman measurement updates.
        """
        # First execute prior diffusion step
        self.predict_step()

        if not feedback_items:
            return {
                "trace_cov": float(np.trace(self.cov_z)),
                "norm_mean": float(np.linalg.norm(self.mean_z)),
                "det_cov": float(np.linalg.det(self.cov_z)),
            }

        for item in feedback_items:
            h = item.horizon
            if h not in self.horizon_to_w:
                continue

            w_h = self.horizon_to_w[h]
            loc = getattr(item, "base_location", None)
            if loc is None:
                loc = float(np.median(item.base_samples))
            residual = float(item.y_true - loc)

            # Expected observation: pred_e = w_h^T m
            pred_e = float(np.dot(w_h, self.mean_z))
            innovation = residual - pred_e

            # Innovation variance: S = w_h^T Cov w_h + R
            s_var = float(w_h.T @ self.cov_z @ w_h + self.r_noise)

            # Kalman gain: K = Cov @ w_h / S
            kalman_gain = (self.cov_z @ w_h) / s_var

            # State update
            self.mean_z = self.mean_z + kalman_gain * innovation

            # Joseph form covariance update for numerical symmetry & positive definiteness:
            # Cov = (I - K w^T) Cov (I - K w^T)^T + K R K^T
            i_kh = np.eye(self.latent_dim) - np.outer(kalman_gain, w_h)
            self.cov_z = i_kh @ self.cov_z @ i_kh.T + np.outer(kalman_gain, kalman_gain) * self.r_noise

            # Update empirical scale
            disp = getattr(item, "base_dispersion", None)
            if disp is None or disp <= 1e-4:
                disp = float(np.std(item.base_samples)) + 1e-4
            emp_scale = abs(residual) / disp
            self.scale_factors[h] = 0.90 * self.scale_factors[h] + 0.10 * max(0.2, min(5.0, emp_scale))

        # Compact projection of mean: ||m|| <= 8.0
        norm_m = float(np.linalg.norm(self.mean_z))
        if norm_m > 8.0:
            self.mean_z = (self.mean_z / norm_m) * 8.0

        return {
            "trace_cov": float(np.trace(self.cov_z)),
            "norm_mean": float(np.linalg.norm(self.mean_z)),
            "det_cov": float(np.linalg.det(self.cov_z)),
        }

    def get_corrections(self, horizon: int) -> Tuple[float, float]:
        """
        Compute location shift and epistemic-hedged scale multiplier.
        """
        w_h = self.horizon_to_w.get(horizon, np.zeros(self.latent_dim))
        loc_shift = float(np.dot(w_h, self.mean_z))

        # Epistemic variance along direction w_h: sigma_epi^2 = w_h^T Cov w_h
        sigma_epi_sq = float(w_h.T @ self.cov_z @ w_h)

        # Scale multiplier inflated by epistemic hedging
        base_s = self.scale_factors.get(horizon, 1.0)
        hedged_scale = base_s * math.sqrt(1.0 + max(0.0, sigma_epi_sq))
        hedged_scale = max(0.1, min(10.0, hedged_scale))

        return loc_shift, hedged_scale

    def get_adapted_samples(
        self,
        horizon: int,
        base_samples: np.ndarray,
        base_location: Optional[float] = None,
        base_dispersion: Optional[float] = None,
    ) -> np.ndarray:
        """Apply Bayesian adapted location and scale to base samples."""
        loc_shift, hedged_scale = self.get_corrections(horizon)

        if base_location is None:
            base_location = float(np.median(base_samples))

        adapted_loc = base_location + loc_shift
        return adapted_loc + hedged_scale * (base_samples - base_location)
