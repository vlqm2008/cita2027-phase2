"""Controlled synthetic distribution-shift benchmarks for multi-horizon adaptation."""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


class SyntheticShiftScenario:
    """Generates synthetic time series and simulated frozen model forecasts across multiple regimes."""

    def __init__(self, horizons: list[int] = (1, 4, 8, 16), n_samples: int = 100, seed: int = 2027):
        self.horizons = sorted(horizons)
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def generate(self, regime: str, length: int = 600) -> dict:
        """Generate true series y and base frozen forecasts for each origin and horizon.

        Returns:
            dict containing:
                - 'y': 1D array of actual values (length T)
                - 'regime': regime name
                - 'horizons': list of horizons
                - 'base_forecasts': dict mapping origin_t -> {h: array(n_samples)}
        """
        rng = self.rng
        t = np.arange(length)
        # Baseline ground truth process: AR(1) with innovations
        phi = 0.5
        noise_std = 1.0
        y = np.zeros(length, dtype=np.float64)

        # Base drift functions depending on regime
        mean_offset = np.zeros(length, dtype=np.float64)
        vol_multiplier = np.ones(length, dtype=np.float64)
        horizon_factors = {h: 1.0 + 0.2 * np.log2(h) for h in self.horizons}

        mid = length // 2

        if regime == "stationary":
            pass

        elif regime == "mean_shift":
            # Structural step shift in mean at mid
            mean_offset[mid:] = 2.0

        elif regime == "variance_shift":
            # Structural volatility jump at mid (2.5x)
            vol_multiplier[mid:] = 2.5

        elif regime == "joint_shift":
            # Both mean jump and volatility jump
            mean_offset[mid:] = 1.8
            vol_multiplier[mid:] = 2.2

        elif regime == "shared_cross_horizon_drift":
            # Shared latent factor Z_t that drives mismatch across all horizons proportionally
            # This is the exact regime where H1 is designed to succeed
            mean_offset[mid:] = 2.5

        elif regime == "horizon_specific_drift":
            # Mismatch at each horizon is uncorrelated / orthogonal
            # H1 should NOT transfer effectively here (test of false positive transfer)
            pass

        elif regime == "abrupt_reversal":
            # Positive shift, followed by sudden sharp reversal
            q1 = length // 3
            q2 = 2 * length // 3
            mean_offset[q1:q2] = 2.5
            mean_offset[q2:] = -2.5

        elif regime == "transient_outliers":
            # Occasional sharp spikes that revert immediately
            outlier_indices = rng.choice(np.arange(mid, length), size=8, replace=False)
            for idx in outlier_indices:
                mean_offset[idx] = rng.choice([-5.0, 5.0])

        else:
            raise ValueError(f"Unknown synthetic regime: {regime}")

        # Simulate realization
        for i in range(1, length):
            eps = rng.normal(0, noise_std * vol_multiplier[i])
            y[i] = phi * y[i - 1] + mean_offset[i] + eps

        # Simulate frozen model forecasts
        # Frozen model is trained on pre-shift stationary dynamics (mean ~ 0, std ~ 1)
        base_forecasts: dict[int, dict[int, np.ndarray]] = {}

        for orig in range(length):
            base_forecasts[orig] = {}
            for h in self.horizons:
                if orig + h >= length:
                    continue

                # The frozen base model predicts assuming stationary AR(1) from current y[orig]
                frozen_pred_mean = (phi ** h) * y[orig]
                frozen_pred_std = noise_std * np.sqrt(np.sum([phi ** (2 * k) for k in range(h)]))

                # If horizon_specific_drift: inject orthogonal distortion per horizon post-shift
                if regime == "horizon_specific_drift" and orig >= mid:
                    h_idx = self.horizons.index(h)
                    # Different sign/amplitude per horizon
                    h_bias = 2.0 * np.sin(h_idx * np.pi / 2.0 + 1.0)
                    samples = rng.normal(frozen_pred_mean + h_bias, frozen_pred_std, size=self.n_samples)
                else:
                    samples = rng.normal(frozen_pred_mean, frozen_pred_std, size=self.n_samples)

                base_forecasts[orig][h] = samples

        return {
            "regime": regime,
            "y": y,
            "horizons": self.horizons,
            "base_forecasts": base_forecasts,
            "length": length,
            "split_point": mid,
        }
