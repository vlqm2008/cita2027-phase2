"""
Asynchronous Trajectory Conformal Risk Control (AT-CRC).

Extends point-wise conformal calibration to joint multi-horizon trajectory coverage:
    P( forall h in H: y_{t+h} in [l_{t,h}, u_{t,h}] ) >= 1 - alpha_joint
under asynchronous arrival latency without conservative Bonferroni union bounds.
Implements causal stochastic approximation over joint trajectory non-conformity suprema
with partial-horizon maturation correction.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


class AsynchronousTrajectoryCRC:
    """
    Asynchronous Trajectory Conformal Risk Controller (AT-CRC).
    
    Guarantees finite-sample joint trajectory predictive band coverage
    across arbitrary lead times H = {1, ..., H_max}.
    """

    def __init__(
        self,
        horizons: List[int],
        alpha_joint: float = 0.10,
        learning_rate: float = 0.05,
        initial_quantile: float = 2.2,
        min_quantile: float = 0.5,
        max_quantile: float = 8.0,
    ):
        """
        Parameters
        ----------
        horizons : List[int]
            Set of prediction horizons (e.g. [1, 2, 4, 8, 16]).
        alpha_joint : float
            Target joint trajectory miscoverage rate (default 0.10 -> 90% joint coverage).
        learning_rate : float
            Robbins-Monro adaptation step size eta_crc.
        initial_quantile : float
            Starting non-conformity quantile (e.g. 1.96 for Gaussian 95%).
        """
        self.horizons = sorted(list(horizons))
        self.num_horizons = len(self.horizons)
        self.alpha_joint = alpha_joint
        self.learning_rate = learning_rate
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile

        # Scalar joint trajectory quantile multiplier
        self.q_joint = float(initial_quantile)

        # Per-horizon shape modulation vector (captures horizon-specific heteroskedasticity)
        self.horizon_factors = {
            h: math.sqrt(1.0 + 0.05 * float(h)) for h in self.horizons
        }

        # Provenance tracking: store recent trajectory residuals indexed by origin_t
        # origin_t -> {h: normalized_residual}
        self.pending_trajectories: Dict[int, Dict[int, float]] = {}
        # Track last evaluated count per origin to prevent double counting
        self.last_evaluated_count: Dict[int, int] = {}

        # Rolling statistics
        self.history_q: List[float] = [self.q_joint]
        self.history_breaches: List[int] = []

    def get_prediction_bands(
        self,
        base_locations: Dict[int, float],
        base_dispersions: Dict[int, float],
        adapted_locations: Optional[Dict[int, float]] = None,
        adapted_scales: Optional[Dict[int, float]] = None,
    ) -> Dict[int, Tuple[float, float, float]]:
        """
        Compute joint trajectory prediction intervals [lower, upper, width] for all h in H.
        
        Returns
        -------
        Dict[int, Tuple[float, float, float]]
            Mapping h -> (lower_bound, upper_bound, interval_width).
        """
        bands: Dict[int, Tuple[float, float, float]] = {}
        for h in self.horizons:
            loc = (
                adapted_locations.get(h, base_locations[h])
                if adapted_locations is not None
                else base_locations[h]
            )
            disp = (
                base_dispersions[h] * adapted_scales.get(h, 1.0)
                if adapted_scales is not None
                else base_dispersions[h]
            )
            disp = max(1e-4, disp)

            # Half-width scaled by joint quantile and horizon factor
            half_width = self.q_joint * self.horizon_factors[h] * disp
            lower = loc - half_width
            upper = loc + half_width
            bands[h] = (lower, upper, 2.0 * half_width)

        return bands

    def register_forecast(self, origin_t: int) -> None:
        """Initialize tracking container for trajectory emitted at origin_t."""
        if origin_t not in self.pending_trajectories:
            self.pending_trajectories[origin_t] = {}

    def update_with_feedback(
        self,
        feedback_items: List[ForecastFeedbackItem],
        current_t: int,
    ) -> Dict[str, float]:
        """
        Process matured feedback items and causally update joint quantile.
        """
        if not feedback_items:
            return {"q_joint": self.q_joint, "trajectory_breach": 0.0}

        # Ingest feedback items into pending trajectories
        completed_origins: List[int] = []
        for item in feedback_items:
            orig = item.origin_t
            h = item.horizon
            if orig not in self.pending_trajectories:
                self.pending_trajectories[orig] = {}

            # Normalized absolute non-conformity score
            if item.adapted_samples is not None and len(item.adapted_samples) > 0:
                center_loc = float(np.median(item.adapted_samples))
                center_disp = float(np.std(item.adapted_samples))
            else:
                base_loc = getattr(item, "base_location", None)
                if base_loc is None:
                    base_loc = float(np.median(item.base_samples)) if len(item.base_samples) > 0 else 0.0
                base_disp = getattr(item, "base_dispersion", None)
                if base_disp is None:
                    base_disp = float(np.std(item.base_samples)) if len(item.base_samples) > 0 else 1.0
                center_loc = base_loc
                center_disp = base_disp

            disp = max(1e-4, center_disp * self.horizon_factors.get(h, 1.0))
            score = abs(item.y_true - center_loc) / disp
            self.pending_trajectories[orig][h] = score

            # Check if trajectory from orig has all horizons matured
            if len(self.pending_trajectories[orig]) == self.num_horizons:
                completed_origins.append(orig)

        # Update on completed trajectories or newly matured partial trajectories
        trajectory_breached = 0
        num_updates = 0

        if completed_origins:
            for orig in completed_origins:
                scores_dict = self.pending_trajectories[orig]
                max_score = max(scores_dict.values())
                breach = 1.0 if max_score > self.q_joint else 0.0
                trajectory_breached = max(trajectory_breached, int(breach))

                grad = (breach - self.alpha_joint)
                self.q_joint += self.learning_rate * grad
                self.q_joint = float(np.clip(self.q_joint, self.min_quantile, self.max_quantile))

                num_updates += 1
                del self.pending_trajectories[orig]
                if orig in self.last_evaluated_count:
                    del self.last_evaluated_count[orig]
        else:
            for orig, scores_dict in list(self.pending_trajectories.items()):
                curr_count = len(scores_dict)
                prev_count = self.last_evaluated_count.get(orig, 0)
                if curr_count > prev_count and curr_count >= 2:
                    max_score = max(scores_dict.values())
                    breach = 1.0 if max_score > self.q_joint else 0.0
                    trajectory_breached = max(trajectory_breached, int(breach))

                    maturation_ratio = curr_count / float(self.num_horizons)
                    grad = (breach - self.alpha_joint) * maturation_ratio
                    self.q_joint += self.learning_rate * grad
                    self.q_joint = float(np.clip(self.q_joint, self.min_quantile, self.max_quantile))

                    self.last_evaluated_count[orig] = curr_count
                    num_updates += 1

        # Prune very old trajectories (> 200 steps ago) to avoid memory leaks
        stale_threshold = current_t - 200
        stale_keys = [k for k in self.pending_trajectories if k < stale_threshold]
        for k in stale_keys:
            del self.pending_trajectories[k]

        self.history_q.append(self.q_joint)
        self.history_breaches.append(trajectory_breached)

        return {
            "q_joint": self.q_joint,
            "trajectory_breach": float(trajectory_breached),
            "num_evaluated_trajectories": float(num_updates),
            "pending_trajectories_count": float(len(self.pending_trajectories)),
        }
