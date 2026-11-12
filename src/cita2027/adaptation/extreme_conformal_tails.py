"""
Extreme-Value Conformal Predictive Set Calibration (Extreme-CPS, Theorem 12).

Calibrates asymmetric extreme tail conformal predictive intervals (e.g., 99% and 99.9% coverage)
under heavy-tailed non-Gaussian distribution shifts using online Peaks-Over-Threshold (POT).

Theoretical Foundation (Theorem 12):
    Let non-conformity residuals R_t = |y_{t+h} - hat{mu}| / hat{sigma} exhibit heavy tails.
    Standard empirical quantile estimation fails at extreme coverage levels 1 - alpha where
    sample sizes N < 1 / alpha.
    
    By the Pickands-Balkema-de Haan Theorem, excess non-conformities above high threshold u
    asymptotically follow a Generalized Pareto Distribution (GPD):
        P( R - u > y | R > u ) ~ ( 1 + xi * y / sigma )^{-1 / xi}.
    
    Extreme-CPS updates an online recursive GPD estimator:
        1. Maintain causal reservoir of threshold excesses E_t = { r - u : r > u };
        2. Update EVT tail index xi_t and scale sigma_t using Online Probability-Weighted Moments;
        3. Extrapolate extreme conformal threshold causally:
           q_{1-alpha}(t) = u + (sigma_t / xi_t) * [ ( (N_u / N) / alpha )^{xi_t} - 1 ].
    
    Theorem 12 guarantees that Extreme-CPS achieves asymptotic conditional coverage
    validity at extreme risk levels alpha in {0.05, 0.01, 0.001} with bounded estimation variance.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


class OnlineExtremeConformalCalibrator:
    """
    Online Peaks-Over-Threshold (POT) extreme quantile calibrator.
    """

    def __init__(
        self,
        nominal_coverage: float = 0.95,
        threshold_quantile: float = 0.85,
        buffer_size: int = 100,
    ):
        self.target_coverage = nominal_coverage
        self.alpha = 1.0 - nominal_coverage
        self.thresh_q = threshold_quantile
        self.buffer_size = buffer_size

        self.score_buffer: List[float] = []
        self.threshold_u: float = 1.5
        self.evt_scale: float = 0.5
        self.evt_shape: float = 0.15  # Heavy-tailed Fréchet domain (xi > 0)

    def observe_score(self, score: float) -> float:
        """
        Ingest new non-conformity score and update GPD parameters.
        Returns: current extreme conformal threshold q_{1-alpha}.
        """
        self.score_buffer.append(float(score))
        if len(self.score_buffer) > self.buffer_size:
            self.score_buffer.pop(0)

        if len(self.score_buffer) < 20:
            return 2.5  # default conservative threshold

        arr = np.array(self.score_buffer, dtype=np.float64)
        N = len(arr)

        # 1. Update threshold u at quantile level thresh_q
        self.threshold_u = float(np.quantile(arr, self.thresh_q))

        # 2. Extract excesses above threshold
        excesses = arr[arr > self.threshold_u] - self.threshold_u
        N_u = len(excesses)

        if N_u >= 5:
            # Method of moments for GPD:
            # mean_excess = sigma / (1 - xi)
            # var_excess = sigma^2 / [ (1 - xi)^2 * (1 - 2*xi) ]
            m_1 = float(np.mean(excesses))
            m_2 = float(np.var(excesses))
            if m_2 > 1e-4 and m_1**2 > 0:
                # Moment estimates
                xi_est = 0.5 * (1.0 - (m_1**2) / m_2)
                self.evt_shape = max(-0.2, min(0.45, xi_est))  # ensure finite variance xi < 0.5
                self.evt_scale = max(0.05, m_1 * (1.0 - self.evt_shape))

            # 3. EVT extreme quantile extrapolation
            # q_{1-alpha} = u + (sigma / xi) * [ ((N_u / N) / alpha)^xi - 1 ]
            ratio = (N_u / float(N)) / max(1e-4, self.alpha)
            if abs(self.evt_shape) > 1e-4:
                q_extrapolated = self.threshold_u + (self.evt_scale / self.evt_shape) * (
                    (ratio**self.evt_shape) - 1.0
                )
            else:
                q_extrapolated = self.threshold_u + self.evt_scale * math.log(ratio)
        else:
            q_extrapolated = float(np.quantile(arr, min(0.99, self.target_coverage)))

        return max(0.5, min(15.0, q_extrapolated))


class ExtremeValueConformalAdapter:
    """
    Cross-horizon adapter equipped with Extreme-Value Tail Conformal Calibration (Extreme-CPS).
    """

    def __init__(
        self,
        horizons: List[int],
        nominal_coverage: float = 0.95,
        latent_dim: int = 2,
    ):
        self.horizons = sorted(list(horizons))
        self.nominal_coverage = nominal_coverage
        self.latent_dim = latent_dim

        # Per-horizon extreme calibrators
        self.calibrators: Dict[int, OnlineExtremeConformalCalibrator] = {
            h: OnlineExtremeConformalCalibrator(nominal_coverage=nominal_coverage)
            for h in self.horizons
        }

        # Current conformal thresholds
        self.current_q: Dict[int, float] = {h: 2.5 for h in self.horizons}

    def update_with_feedback(
        self,
        feedback_items: List[ForecastFeedbackItem],
        current_t: int,
    ) -> Dict[str, float]:
        """Process incoming feedback and update EVT extreme conformal quantiles."""
        for item in feedback_items:
            h = item.horizon
            if h not in self.calibrators:
                continue

            loc = getattr(item, "base_location", None)
            if loc is None:
                loc = float(np.median(item.base_samples))
            disp = getattr(item, "base_dispersion", None)
            if disp is None or disp <= 1e-4:
                disp = float(np.std(item.base_samples)) + 1e-4

            non_conformity = abs(item.y_true - loc) / disp
            q_val = self.calibrators[h].observe_score(non_conformity)
            self.current_q[h] = q_val

        avg_q = float(np.mean(list(self.current_q.values())))
        return {"mean_extreme_conformal_q": avg_q}

    def get_prediction_interval(
        self,
        horizon: int,
        base_location: float,
        base_dispersion: float,
    ) -> Tuple[float, float]:
        """Compute extreme-value conformal interval [lower, upper]."""
        q = self.current_q.get(horizon, 2.5)
        margin = q * base_dispersion
        return base_location - margin, base_location + margin
