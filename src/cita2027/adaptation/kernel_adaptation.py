"""
Nonlinear Kernelized Cross-Horizon Probabilistic Adaptation (Kernel-H3).

Extends the linear representation manifold to an infinite-dimensional Reproducing Kernel
Hilbert Space (RKHS) via Random Fourier Features (RFF) (Rahimi & Recht, 2007).
Captures complex nonlinear cross-horizon interactions (e.g. yield curve twisting,
butterfly shifts, and nonlinear volatility leverage) while maintaining strict O(D)
computational complexity and causal prequential execution.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


class RandomFourierFeaturesEmbedding:
    """
    Random Fourier Features (RFF) projector for cross-horizon embeddings.
    
    Approximates the Gaussian RBF kernel:
        k(x_i, x_j) = exp(- ||x_i - x_j||^2 / (2 * sigma_k^2))
    via explicit randomized mapping:
        phi(x) = sqrt(2 / D) * cos(Omega * x + b)
    where Omega ~ N(0, 1 / sigma_k^2) and b ~ Uniform[0, 2*pi].
    """

    def __init__(
        self,
        input_dim: int = 3,
        num_features: int = 16,
        kernel_gamma: float = 1.0,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of input features (e.g. log2(h), recent_vol, recent_drift).
        num_features : int
            Number of random Fourier features (D).
        kernel_gamma : float
            RBF kernel parameter gamma = 1 / (2 * sigma_k^2).
        seed : int
            Random seed for deterministic feature matrix initialization.
        """
        self.input_dim = input_dim
        self.num_features = num_features
        self.kernel_gamma = kernel_gamma
        self.seed = seed

        rng = np.random.default_rng(seed)
        sigma = math.sqrt(2.0 * kernel_gamma)
        self.omega = rng.normal(loc=0.0, scale=sigma, size=(num_features, input_dim))
        self.b = rng.uniform(low=0.0, high=2.0 * math.pi, size=num_features)
        self.scale_factor = math.sqrt(2.0 / num_features)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Map input vector x in R^input_dim to phi(x) in R^num_features.
        """
        x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
        if len(x_vec) != self.input_dim:
            raise ValueError(f"Expected input of dim {self.input_dim}, got {len(x_vec)}")
        projection = np.dot(self.omega, x_vec) + self.b
        return self.scale_factor * np.cos(projection)


class HypothesisKernelH3_Corrector:
    """
    Nonlinear Kernelized Online Cross-Horizon Corrector (Kernel-H3).
    
    Adapts multi-horizon predictive distributions via online gradient descent
    in the Random Fourier Feature Hilbert space with Kish effective-sample-size
    overlap regularized step sizes and exponential age decay.
    """

    def __init__(
        self,
        horizons: List[int],
        rff_dim: int = 16,
        learning_rate: float = 0.05,
        age_decay_gamma: float = 0.08,
        overlap_discount_beta: float = 0.5,
        l2_regularization: float = 1e-4,
        max_state_norm: float = 8.0,
        step_clip: float = 1.5,
        seed: int = 42,
    ):
        self.horizons = sorted(list(horizons))
        self.rff_dim = rff_dim
        self.learning_rate = learning_rate
        self.age_decay_gamma = age_decay_gamma
        self.overlap_discount_beta = overlap_discount_beta
        self.l2_reg = l2_regularization
        self.max_state_norm = max_state_norm
        self.step_clip = step_clip

        # Input feature: [log2(h) / 4.0, estimated_volatility, drift_sign]
        self.rff = RandomFourierFeaturesEmbedding(
            input_dim=3,
            num_features=rff_dim,
            kernel_gamma=1.5,
            seed=seed,
        )

        # Online weights in R^rff_dim
        self.w_mean = np.zeros(rff_dim, dtype=np.float64)
        self.w_scale = np.zeros(rff_dim, dtype=np.float64)
        self.local_bias = {h: 0.0 for h in self.horizons}
        self.local_log_scale = {h: 0.0 for h in self.horizons}

        # Track contextual volatility and recent drift
        self.recent_vol = 1.0
        self.recent_residual = 0.0

    def _get_context_vector(self, h: int) -> np.ndarray:
        norm_h = math.log2(max(1, h)) / 4.0
        vol_feat = float(np.clip(self.recent_vol, 0.1, 5.0))
        drift_feat = float(np.clip(self.recent_residual, -3.0, 3.0))
        return np.array([norm_h, vol_feat, drift_feat], dtype=np.float64)

    def get_corrections(self, h: int) -> Tuple[float, float]:
        """
        Compute adapted mean shift Delta mu and scale multiplier s for horizon h.
        """
        x_ctx = self._get_context_vector(h)
        phi_h = self.rff.transform(x_ctx)

        # Nonlinear mean shift
        delta_mu = float(np.dot(phi_h, self.w_mean) + self.local_bias.get(h, 0.0))

        # Nonlinear log-scale multiplier
        log_scale = float(np.dot(phi_h, self.w_scale) + self.local_log_scale.get(h, 0.0))
        log_scale = float(np.clip(log_scale, -1.6, 1.6))
        scale_mult = float(math.exp(log_scale))

        return delta_mu, scale_mult

    def adapt_samples(
        self,
        base_samples: np.ndarray,
        base_location: float,
        h: int,
    ) -> np.ndarray:
        """
        Apply nonlinear location-scale transformation to predictive samples.
        """
        delta_mu, scale_mult = self.get_corrections(h)
        adapted = (base_location + delta_mu) + scale_mult * (base_samples - base_location)
        return adapted

    def update_with_feedback(
        self,
        feedback_items: List[ForecastFeedbackItem],
        current_t: int,
    ) -> Dict[str, float]:
        """
        Update kernel manifold weights using matured feedback batch.
        """
        if not feedback_items:
            return {"num_items": 0, "effective_k": 0.0, "step_norm": 0.0}

        k = len(feedback_items)
        raw_weights: List[float] = []
        grad_mean_list: List[np.ndarray] = []
        grad_scale_list: List[np.ndarray] = []

        for item in feedback_items:
            h = item.horizon
            age = max(0, current_t - item.origin_t)
            age_weight = math.exp(-self.age_decay_gamma * age)

            base_loc = getattr(item, "base_location", None)
            if base_loc is None:
                base_loc = float(np.median(item.base_samples)) if len(item.base_samples) > 0 else 0.0

            base_disp = getattr(item, "base_dispersion", None)
            if base_disp is None:
                base_disp = float(np.std(item.base_samples)) if len(item.base_samples) > 0 else 1.0
            base_disp = max(1e-4, base_disp)

            # Causal residual against current adapted forecast
            delta_mu, scale_mult = self.get_corrections(h)
            adapted_loc = base_loc + delta_mu
            residual = item.y_true - adapted_loc

            # Drift consistency gating
            alignment = 1.0
            if abs(self.recent_residual) > 1e-4 and abs(residual) > 1e-4:
                cos_sim = (self.recent_residual * residual) / (
                    abs(self.recent_residual) * abs(residual)
                )
                alignment = max(0.1, float(cos_sim))

            w_comp = age_weight * alignment
            raw_weights.append(w_comp)

            # Compute RFF gradient
            x_ctx = self._get_context_vector(h)
            phi_h = self.rff.transform(x_ctx)

            grad_mean = -residual * phi_h
            grad_mean_list.append(grad_mean)

            # Scale gradient based on dispersion mismatch
            est_std = max(1e-3, base_disp * scale_mult)
            scale_error = (residual ** 2) / (est_std ** 2) - 1.0
            scale_error = float(np.clip(scale_error, -2.0, 2.0))
            grad_scale = -scale_error * phi_h
            grad_scale_list.append(grad_scale)

            # Local bias/scale leak update
            self.local_bias[h] = float(
                np.clip(self.local_bias[h] + 0.02 * w_comp * residual, -3.0, 3.0)
            )
            self.local_log_scale[h] = float(
                np.clip(self.local_log_scale[h] + 0.01 * w_comp * scale_error, -1.0, 1.0)
            )

            # Update context tracking
            self.recent_residual = 0.9 * self.recent_residual + 0.1 * residual
            self.recent_vol = 0.95 * self.recent_vol + 0.05 * abs(residual)

        # Kish effective sample size
        sum_w = sum(raw_weights)
        sum_sq = sum(w ** 2 for w in raw_weights)
        k_eff = (sum_w ** 2) / (sum_sq + 1e-8)

        overlap_factor = (k_eff / k) ** self.overlap_discount_beta
        eta_eff = self.learning_rate * overlap_factor

        # Aggregated weighted gradient
        agg_grad_mean = np.zeros(self.rff_dim, dtype=np.float64)
        agg_grad_scale = np.zeros(self.rff_dim, dtype=np.float64)
        for w, gm, gs in zip(raw_weights, grad_mean_list, grad_scale_list):
            agg_grad_mean += w * gm
            agg_grad_scale += w * gs

        if sum_w > 1e-8:
            agg_grad_mean /= sum_w
            agg_grad_scale /= sum_w

        # L2 weight decay regularizer
        agg_grad_mean += self.l2_reg * self.w_mean
        agg_grad_scale += self.l2_reg * self.w_scale

        # Step clip
        step_mean = -eta_eff * agg_grad_mean
        step_scale = -eta_eff * agg_grad_scale

        norm_m = float(np.linalg.norm(step_mean))
        if norm_m > self.step_clip:
            step_mean = (step_mean / norm_m) * self.step_clip

        norm_s = float(np.linalg.norm(step_scale))
        if norm_s > self.step_clip:
            step_scale = (step_scale / norm_s) * self.step_clip

        self.w_mean += step_mean
        self.w_scale += step_scale

        # Projection onto compact ball
        total_norm = float(np.linalg.norm(self.w_mean))
        if total_norm > self.max_state_norm:
            self.w_mean = (self.w_mean / total_norm) * self.max_state_norm

        total_norm_s = float(np.linalg.norm(self.w_scale))
        if total_norm_s > self.max_state_norm:
            self.w_scale = (self.w_scale / total_norm_s) * self.max_state_norm

        return {
            "num_items": k,
            "effective_k": float(k_eff),
            "step_norm": float(norm_m),
            "state_norm_mean": float(np.linalg.norm(self.w_mean)),
            "state_norm_scale": float(np.linalg.norm(self.w_scale)),
        }
