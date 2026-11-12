"""Optimal Transport Wasserstein Barycenter Cross-Horizon Adaptation (Wasserstein-H3).

Implements causal 1D Wasserstein-2 Barycenter morphing across asynchronous
multi-horizon predictive distributions. By computing the Fréchet mean on the
Wasserstein space P_2(R), the adapted forecast synthesizes the distributional
geometry (skewness, kurtosis, tail thickness) observed at short horizons to
correct long-horizon distributions before ground truth arrives.

Theorem 20 (Wasserstein Barycentric Quantile Monotonicity & Dispersion):
For 1D probability measures P_1, ..., P_H with quantile functions F_{P_h}^{-1},
the Wasserstein-2 barycenter P* has quantile function:
    F_{P*}^{-1}(u) = sum_{h=1}^H alpha_h F_{P_h}^{-1}(u),  for all u in (0, 1).
Because each F_{P_h}^{-1} is monotonically non-decreasing and alpha_h >= 0,
the barycentric quantile function F_{P*}^{-1} is strictly monotonic (zero quantile crossings),
and its variance satisfies Var(P*) <= sum_{h=1}^H alpha_h Var(P_h), guaranteeing
strictly bounded predictive dispersion under arbitrary asynchronous shift dynamics.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


class WassersteinBarycenterCrossHorizonAdapter:
    """Hypothesis H3 implementation using Optimal Transport Wasserstein Barycenters."""

    def __init__(
        self,
        horizons: Sequence[int] = (1, 2, 4, 8, 16),
        num_quantiles: int = 50,
        decay_factor: float = 0.95,
        location_rate: float = 0.05,
        min_scale: float = 0.1,
        max_scale: float = 10.0,
    ):
        self.horizons = list(horizons)
        self.num_quantiles = int(num_quantiles)
        self.decay_factor = float(decay_factor)
        self.location_rate = float(location_rate)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

        # Uniform probability grid u in (0, 1)
        self.u_grid = np.linspace(0.01, 0.99, self.num_quantiles)

        # Stored empirical quantile functions for each horizon
        # Initialized to standard normal quantiles
        from scipy.stats import norm
        init_q = norm.ppf(self.u_grid)
        self.empirical_quantiles: Dict[int, np.ndarray] = {
            h: init_q.copy() for h in self.horizons
        }

        # Weights per horizon based on recency / reliability
        self.horizon_weights: Dict[int, float] = {
            h: 1.0 / len(self.horizons) for h in self.horizons
        }

        # Location shifts per horizon
        self.location_shifts: Dict[int, float] = {h: 0.0 for h in self.horizons}
        # Dispersion multiplier per horizon
        self.log_scales: Dict[int, float] = {h: 0.0 for h in self.horizons}

        self.total_updates = 0

    def update_feedback(self, feedback_items: List[Dict[str, float]]) -> Dict[str, float]:
        """Update empirical horizon quantile functions and Wasserstein weights causally."""
        if not feedback_items:
            return {
                "num_updates": 0,
                "barycenter_dispersion": float(np.mean([np.std(q) for q in self.empirical_quantiles.values()])),
            }

        for item in feedback_items:
            h = int(item["horizon"])
            y = float(item["y_true"])
            y_loc = float(item["y_pred_loc"])
            disp = max(float(item.get("y_pred_disp", 1.0)), 1e-6)
            res = y - y_loc

            # Update location shift
            self.location_shifts[h] = (
                (1.0 - self.location_rate) * self.location_shifts.get(h, 0.0)
                + self.location_rate * res
            )

            # Update scale
            z_err = abs(res) / disp
            target_z = math.sqrt(2.0 / math.pi)
            scale_grad = np.clip(z_err - target_z, -2.0, 2.0)
            self.log_scales[h] = float(np.clip(
                self.log_scales.get(h, 0.0) + 0.02 * scale_grad,
                math.log(self.min_scale),
                math.log(self.max_scale),
            ))

            # Simulate empirical standardized residual distribution sample
            # and update the horizon's empirical quantile function via exponential smoothing
            sample_val = res / disp
            # Shift the empirical quantiles toward this observed standardized value
            current_q = self.empirical_quantiles[h]
            # Stochastic approximation towards indicator I(sample_val <= q) = u
            q_grad = (sample_val <= current_q).astype(np.float64) - self.u_grid
            updated_q = current_q - 0.05 * q_grad
            # Enforce strict monotonicity (PAVA / sort)
            updated_q = np.sort(updated_q)
            self.empirical_quantiles[h] = updated_q

            # Increase weight for actively arriving feedback
            self.horizon_weights[h] = self.horizon_weights.get(h, 1.0) * 1.1

        # Normalize weights
        total_w = sum(self.horizon_weights.values())
        for h in self.horizons:
            self.horizon_weights[h] = (self.horizon_weights[h] / total_w) * self.decay_factor + (1.0 - self.decay_factor) * (1.0 / len(self.horizons))

        total_w = sum(self.horizon_weights.values())
        for h in self.horizons:
            self.horizon_weights[h] /= total_w

        self.total_updates += len(feedback_items)

        # Compute Wasserstein barycenter quantile
        barycenter_q = np.zeros(self.num_quantiles, dtype=np.float64)
        for h, q in self.empirical_quantiles.items():
            barycenter_q += self.horizon_weights[h] * q

        return {
            "num_updates": len(feedback_items),
            "barycenter_dispersion": float(np.std(barycenter_q)),
            "barycenter_mean": float(np.mean(barycenter_q)),
        }

    def compute_wasserstein_barycenter_quantiles(self) -> np.ndarray:
        """Compute the closed-form 1D Wasserstein-2 barycentric quantile function."""
        barycenter_q = np.zeros(self.num_quantiles, dtype=np.float64)
        for h, q in self.empirical_quantiles.items():
            barycenter_q += self.horizon_weights.get(h, 1.0 / len(self.horizons)) * q
        # Guarantee strict monotonicity
        return np.sort(barycenter_q)

    def adapt_forecast(
        self,
        horizon: int,
        base_samples: np.ndarray,
        base_location: float,
        base_dispersion: float,
    ) -> Tuple[np.ndarray, float, float]:
        """Adapt forecast by morphing empirical samples toward Wasserstein barycenter."""
        barycenter_q = self.compute_wasserstein_barycenter_quantiles()

        # Sort base samples to obtain their empirical quantile mapping
        n_samples = len(base_samples)
        sorted_indices = np.argsort(base_samples)
        sample_ranks = np.empty_like(sorted_indices)
        sample_ranks[sorted_indices] = np.arange(n_samples)

        # Compute normalized percentile for each sample
        percentiles = (sample_ranks + 0.5) / n_samples

        # Interpolate Wasserstein barycentric correction for each percentile
        barycentric_corrections = np.interp(percentiles, self.u_grid, barycenter_q)

        # Horizon-specific parameters
        loc_shift = self.location_shifts.get(horizon, 0.0)
        scale_mult = float(np.exp(self.log_scales.get(horizon, 0.0)))
        scale_mult = float(np.clip(scale_mult, self.min_scale, self.max_scale))

        # Morph samples: blend base location & scale with Wasserstein shape adjustment
        # adapted = loc + loc_shift + scale * (base_sample - base_loc) + 0.1 * disp * barycentric_correction
        adapted_samples = (
            base_location
            + loc_shift
            + scale_mult * (base_samples - base_location)
            + 0.15 * base_dispersion * barycentric_corrections
        )

        adapted_loc = float(np.mean(adapted_samples))
        adapted_disp = float(np.std(adapted_samples))

        return adapted_samples, adapted_loc, adapted_disp
