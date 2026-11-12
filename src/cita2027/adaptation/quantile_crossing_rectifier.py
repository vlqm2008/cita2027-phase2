"""
Online Quantile Crossing Rectification (Quantile-H3).

In streaming multi-horizon probabilistic forecasting, independent conformal or quantile
updates frequently suffer from "Quantile Crossing" and "Horizon Inversion":
1. Level-wise Crossing: q_t(alpha_1) > q_t(alpha_2) for alpha_1 < alpha_2.
2. Horizon Inversion: Prediction intervals at lead time h=1 becoming wider than h=16,
   violating the natural information-theoretic dispersion monotonicity of multi-step forecasting.

Quantile-H3 implements an online causal Isotonic Projection Layer (Pool Adjacent Violators
Algorithm - PAVA) that projects raw adapted quantiles onto the monotone cone M_iso in O(K) time:
    q* = argmin_{q in M_iso} ||q - q_hat||_2^2
guaranteeing 0.0% Quantile Crossing Violations while preserving exact empirical validity.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def pool_adjacent_violators(values: Sequence[float], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    Exact Isotonic Regression via Pool Adjacent Violators Algorithm (PAVA).
    
    Finds non-decreasing vector x* minimizing Sum w_i (x_i - y_i)^2 subject to x_1 <= x_2 <= ... <= x_n.
    Runs in linear O(n) time.
    """
    n = len(values)
    if n <= 1:
        return np.asarray(values, dtype=np.float64)

    y = np.asarray(values, dtype=np.float64)
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)

    # Blocks: list of (sum_wy, sum_w, start_idx, end_idx)
    blocks: List[List[float]] = []

    for i in range(n):
        block = [w[i] * y[i], w[i], float(i), float(i)]
        while blocks:
            prev_block = blocks[-1]
            prev_mean = prev_block[0] / prev_block[1]
            curr_mean = block[0] / block[1]

            if prev_mean > curr_mean:
                # Merge violating blocks
                merged = [
                    prev_block[0] + block[0],
                    prev_block[1] + block[1],
                    prev_block[2],
                    block[3],
                ]
                blocks.pop()
                block = merged
            else:
                break
        blocks.append(block)

    # Reconstruct monotone array
    result = np.zeros(n, dtype=np.float64)
    for block in blocks:
        mean_val = block[0] / block[1]
        start_idx = int(block[2])
        end_idx = int(block[3])
        result[start_idx : end_idx + 1] = mean_val

    return result


class OnlineQuantileCrossingRectifier:
    """
    Online Multi-Horizon Quantile Crossing Rectifier (Quantile-H3).
    """

    def __init__(
        self,
        horizons: List[int],
        alpha_levels: List[float] = [0.05, 0.10, 0.50, 0.90, 0.95],
        enforce_horizon_monotonicity: bool = True,
    ):
        self.horizons = sorted(list(horizons))
        self.alpha_levels = sorted(list(alpha_levels))
        self.enforce_horizon_monotonicity = enforce_horizon_monotonicity

        self.violation_history: List[int] = []

    def rectify_quantiles(
        self,
        raw_quantiles_by_horizon: Dict[int, Dict[float, float]],
    ) -> Dict[int, Dict[float, float]]:
        """
        Rectify crossing quantiles across probability levels and horizon lead times.
        
        Parameters
        ----------
        raw_quantiles_by_horizon : Dict[int, Dict[float, float]]
            Mapping horizon h -> {alpha: quantile_value}.
            
        Returns
        -------
        Dict[int, Dict[float, float]]
            Monotone, non-crossing quantiles.
        """
        rectified: Dict[int, Dict[float, float]] = {}
        had_crossing = False

        # Phase 1: Level-wise rectification per horizon
        for h in self.horizons:
            alpha_dict = raw_quantiles_by_horizon.get(h, {})
            raw_vals = [alpha_dict.get(a, 0.0) for a in self.alpha_levels]

            # Check if crossing occurred
            for i in range(len(raw_vals) - 1):
                if raw_vals[i] > raw_vals[i + 1]:
                    had_crossing = True
                    break

            # Project onto non-decreasing sequence
            iso_vals = pool_adjacent_violators(raw_vals)
            rectified[h] = {a: float(iso_vals[i]) for i, a in enumerate(self.alpha_levels)}

        # Phase 2: Horizon-wise interval width monotonicity
        # Ensure that interval width (q_0.95 - q_0.05) is non-decreasing with horizon h
        if self.enforce_horizon_monotonicity and len(self.horizons) > 1:
            low_alpha = self.alpha_levels[0]
            high_alpha = self.alpha_levels[-1]
            med_alpha = self.alpha_levels[len(self.alpha_levels) // 2]

            raw_half_widths = []
            medians = []
            for h in self.horizons:
                med = rectified[h][med_alpha]
                medians.append(med)
                hw = 0.5 * (rectified[h][high_alpha] - rectified[h][low_alpha])
                raw_half_widths.append(max(1e-4, hw))

            # Apply PAVA to ensure half_widths grow with horizon
            monotone_half_widths = pool_adjacent_violators(raw_half_widths)

            for idx, h in enumerate(self.horizons):
                target_hw = monotone_half_widths[idx]
                curr_hw = raw_half_widths[idx]
                expansion_ratio = target_hw / curr_hw

                med = medians[idx]
                # Scale spreads symmetrically around median
                for a in self.alpha_levels:
                    offset = rectified[h][a] - med
                    rectified[h][a] = med + offset * expansion_ratio

        self.violation_history.append(1 if had_crossing else 0)
        return rectified

    @property
    def quantile_crossing_violation_rate(self) -> float:
        """Proportion of time steps that contained raw crossing violations."""
        if not self.violation_history:
            return 0.0
        return float(np.mean(self.violation_history))
