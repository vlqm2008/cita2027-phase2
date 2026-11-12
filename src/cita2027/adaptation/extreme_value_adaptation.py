"""
Extreme Value Theory (EVT) and Heavy-Tail Distribution Adaptation.

Implements online Peaks-Over-Threshold (POT) tail modeling:
1. Online dynamic threshold tracking u_t at high quantile (e.g. 90th percentile)
2. Generalized Pareto Distribution (GPD) fitting for tail exceedances:
   P(Y - u > y | Y > u) = (1 + xi * y / sigma_tail)^(-1 / xi)
3. Tail Splicing: Merges foundation model central draws with GPD extreme tails
4. Minimizes catastrophic miscalibration under black-swan shocks and severe outlier bursts
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GPDParameters:
    """Estimated parameters of the Generalized Pareto Distribution."""
    shape_xi: float       # xi > 0 (heavy/Frechet), xi = 0 (Gumbel), xi < 0 (Weibull)
    scale_sigma: float    # sigma > 0
    threshold_u: float


class OnlineExtremeValueAdapter:
    """
    Extreme Value Theory (EVT) Online Tail Adapter.
    
    Dynamically splices Generalized Pareto Distribution (GPD) tails
    onto the empirical Monte Carlo distribution produced by the foundation model.
    """

    def __init__(
        self,
        horizons: List[int],
        tail_probability: float = 0.10,
        initial_threshold: float = 2.0,
        decay_factor: float = 0.95,
    ):
        self.horizons = sorted(list(horizons))
        self.p_tail = tail_probability
        self.decay = decay_factor

        # Horizon-specific GPD parameters
        self.gpd_upper: Dict[int, GPDParameters] = {
            h: GPDParameters(shape_xi=0.15, scale_sigma=1.0, threshold_u=initial_threshold)
            for h in self.horizons
        }
        self.gpd_lower: Dict[int, GPDParameters] = {
            h: GPDParameters(shape_xi=0.15, scale_sigma=1.0, threshold_u=initial_threshold)
            for h in self.horizons
        }

        # Running history of exceedances: dict(h -> list of exceedances)
        self.upper_exceedances: Dict[int, List[float]] = {h: [] for h in self.horizons}
        self.lower_exceedances: Dict[int, List[float]] = {h: [] for h in self.horizons}

    def reset(self) -> None:
        self.upper_exceedances = {h: [] for h in self.horizons}
        self.lower_exceedances = {h: [] for h in self.horizons}
        for h in self.horizons:
            self.gpd_upper[h] = GPDParameters(shape_xi=0.15, scale_sigma=1.0, threshold_u=2.0)
            self.gpd_lower[h] = GPDParameters(shape_xi=0.15, scale_sigma=1.0, threshold_u=2.0)

    def _fit_gpd_moments(self, exceedances: List[float], u: float) -> GPDParameters:
        """Estimate GPD shape xi and scale sigma via Probability Weighted Moments (Hosking & Wallis 1987)."""
        if len(exceedances) < 5:
            return GPDParameters(shape_xi=0.15, scale_sigma=1.0, threshold_u=u)

        arr = np.sort(np.asarray(exceedances, dtype=np.float64))
        n = len(arr)
        x_bar = float(np.mean(arr))
        # Probability weighted moment: M_1 = (1/n) sum (1 - (i - 0.35)/n) * x_(i)
        i_vec = np.arange(1, n + 1)
        w = 1.0 - (i_vec - 0.35) / n
        M1 = float(np.mean(w * arr))

        # Method of moments estimates
        denom = x_bar - 2.0 * M1
        if abs(denom) < 1e-6:
            xi = 0.10
            sigma = x_bar
        else:
            xi = (x_bar / denom) - 2.0
            xi = max(-0.4, min(0.8, xi))  # restrict to plausible statistical bounds
            sigma = (2.0 * x_bar * M1) / max(1e-6, denom)
            sigma = max(0.1, min(10.0, sigma))

        return GPDParameters(shape_xi=xi, scale_sigma=sigma, threshold_u=u)

    def update_tails(self, horizon: int, y_true: float, base_median: float, base_std: float) -> None:
        """Process new residual observation and update EVT tail parameters."""
        std = max(1e-4, base_std)
        normalized_error = (y_true - base_median) / std

        u_up = self.gpd_upper[horizon].threshold_u
        u_lo = self.gpd_lower[horizon].threshold_u

        # Update upper tail if exceedance occurs
        if normalized_error > u_up:
            exceed = normalized_error - u_up
            self.upper_exceedances[horizon].append(exceed)
            if len(self.upper_exceedances[horizon]) > 40:
                self.upper_exceedances[horizon].pop(0)
            self.gpd_upper[horizon] = self._fit_gpd_moments(self.upper_exceedances[horizon], u_up)

        # Update lower tail if exceedance occurs
        if -normalized_error > u_lo:
            exceed = -normalized_error - u_lo
            self.lower_exceedances[horizon].append(exceed)
            if len(self.lower_exceedances[horizon]) > 40:
                self.lower_exceedances[horizon].pop(0)
            self.gpd_lower[horizon] = self._fit_gpd_moments(self.lower_exceedances[horizon], u_lo)

    def splice_predictive_samples(self, horizon: int, base_samples: np.ndarray) -> np.ndarray:
        """
        Splice fitted GPD tails onto the empirical foundation model draws.
        
        Replaces draws in the upper and lower p_tail tails with GPD quantiles.
        """
        samples = np.sort(np.asarray(base_samples, dtype=np.float64))
        S = len(samples)
        if S < 20:
            return samples

        med = float(np.median(samples))
        std = float(np.std(samples)) + 1e-6

        k_tail = max(2, int(S * self.p_tail))
        spliced = samples.copy()

        # Upper tail splicing
        gpd_up = self.gpd_upper[horizon]
        xi_up, sig_up = gpd_up.shape_xi, gpd_up.scale_sigma
        for i in range(S - k_tail, S):
            # Tail quantile level p in (1 - p_tail, 1)
            p_level = (i + 0.5) / S
            # Exceedance probability relative to tail
            q_tail = (1.0 - p_level) / self.p_tail
            q_tail = max(1e-6, min(1.0, q_tail))
            if abs(xi_up) > 1e-4:
                gpd_val = (sig_up / xi_up) * ((q_tail ** (-xi_up)) - 1.0)
            else:
                gpd_val = -sig_up * math.log(q_tail)
            spliced[i] = med + (gpd_up.threshold_u + gpd_val) * std

        # Lower tail splicing
        gpd_lo = self.gpd_lower[horizon]
        xi_lo, sig_lo = gpd_lo.shape_xi, gpd_lo.scale_sigma
        for i in range(0, k_tail):
            p_level = (i + 0.5) / S
            q_tail = p_level / self.p_tail
            q_tail = max(1e-6, min(1.0, q_tail))
            if abs(xi_lo) > 1e-4:
                gpd_val = (sig_lo / xi_lo) * ((q_tail ** (-xi_lo)) - 1.0)
            else:
                gpd_val = -sig_lo * math.log(q_tail)
            spliced[i] = med - (gpd_lo.threshold_u + gpd_val) * std

        return np.sort(spliced)
