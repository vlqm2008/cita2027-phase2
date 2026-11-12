"""
Competitor Adapters for Online Multi-Horizon Probabilistic Forecasting.

This module provides state-of-the-art delayed and overlapping feedback adaptation
baselines adapted to probabilistic time-series foundation model adjustment:
1. PROCEED (Probabilistic Proactive Concept Drift Adaptation with prototype memory)
2. PADRE (Pattern-Aware Delayed Residual Estimation)
3. StayCALM (Calibrated Adaptive Online Aggregator with overlap discount)
4. AcMCP (Asymmetric Multi-Horizon Online Conformal Prediction)
5. RLS-TS (Recursive Least Squares with Horizon Covariance Tracking)

All competitors conform to the causal prequential protocol:
- No future observation leakage
- Predictions are generated at step t using only observations available at step t
- Delayed feedback arrives when (t - arrival_step >= horizon)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ForecastFeedback:
    """Feedback item emitted when horizon observation matures."""
    origin_step: int
    arrival_step: int
    horizon: int
    y_true: float
    y_pred_samples: np.ndarray
    base_samples: np.ndarray

    @property
    def origin_t(self) -> int:
        return self.origin_step

    @property
    def target_t(self) -> int:
        return self.arrival_step

    @property
    def adapted_samples(self) -> np.ndarray:
        return self.y_pred_samples


class BaseCompetitor:
    """Abstract base class for online adaptation competitors."""

    def __init__(self, name: str, horizons: List[int]):
        self.name = name
        self.horizons = sorted(list(horizons))
        self.step_count = 0

    def predict(
        self,
        horizon: int,
        base_samples: np.ndarray,
        current_step: int,
    ) -> np.ndarray:
        """Adjust base foundation model samples for a specific horizon.
        
        Parameters
        ----------
        horizon : int
            Forecast step ahead (h in H).
        base_samples : np.ndarray
            Array of shape (S,) containing raw Monte Carlo or quantile forecasts.
        current_step : int
            Current global prequential timestep t.
            
        Returns
        -------
        np.ndarray
            Adjusted forecast samples of shape (S,).
        """
        raise NotImplementedError

    def update(
        self,
        feedback_batch: List[ForecastFeedback],
        current_step: int,
    ) -> None:
        """Update internal parameters given newly arrived feedback observations.
        
        Parameters
        ----------
        feedback_batch : List[ForecastFeedback]
            List of feedback observations that matured at current_step.
        current_step : int
            Current global prequential timestep t.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal parameters to initialization state."""
        self.step_count = 0


# ==============================================================================
# 1. PROCEED: Proactive Concept Drift Adaptation with Prototype Memory Buffer
# ==============================================================================

@dataclass
class DriftPrototype:
    """Cluster prototype representing a regime or drift pattern."""
    mean_residual: float
    var_residual: float
    count: int = 1
    last_seen: int = 0


class ProceedProbabilistic(BaseCompetitor):
    """
    PROCEED: Proactive Concept Drift Adaptation adapted for probabilistic TS.
    
    Maintains a bounded memory buffer of recent prediction residuals and
    clusters them into dynamic prototypes. When delayed feedback arrives,
    it identifies the nearest active regime and blends the prototype correction
    with a short-term momentum buffer.
    """

    def __init__(
        self,
        horizons: List[int],
        buffer_capacity: int = 64,
        n_prototypes: int = 4,
        learning_rate: float = 0.15,
        memory_decay: float = 0.95,
    ):
        super().__init__("PROCEED", horizons)
        self.buffer_capacity = buffer_capacity
        self.n_prototypes = n_prototypes
        self.lr = learning_rate
        self.memory_decay = memory_decay

        # Horizon-specific state: location shift and scale multiplier
        self.h_loc: Dict[int, float] = {h: 0.0 for h in self.horizons}
        self.h_scale: Dict[int, float] = {h: 1.0 for h in self.horizons}

        # Memory buffer of residuals: (residual, scale_ratio, age)
        self.memory_buffer: deque = deque(maxlen=buffer_capacity)
        self.prototypes: List[DriftPrototype] = []

    def reset(self) -> None:
        super().reset()
        self.h_loc = {h: 0.0 for h in self.horizons}
        self.h_scale = {h: 1.0 for h in self.horizons}
        self.memory_buffer.clear()
        self.prototypes.clear()

    def _update_prototypes(self, residual: float, variance: float, step: int) -> None:
        if not self.prototypes:
            self.prototypes.append(DriftPrototype(residual, variance, 1, step))
            return

        # Find closest prototype by Euclidean distance
        dists = [abs(p.mean_residual - residual) for p in self.prototypes]
        min_idx = int(np.argmin(dists))

        if dists[min_idx] < 1.0 or len(self.prototypes) >= self.n_prototypes:
            # Merge into closest prototype
            proto = self.prototypes[min_idx]
            eta = 1.0 / (proto.count + 1)
            proto.mean_residual = (1.0 - eta) * proto.mean_residual + eta * residual
            proto.var_residual = (1.0 - eta) * proto.var_residual + eta * variance
            proto.count += 1
            proto.last_seen = step
        else:
            # Spawn new prototype
            self.prototypes.append(DriftPrototype(residual, variance, 1, step))

    def update(self, feedback_batch: List[ForecastFeedback], current_step: int) -> None:
        self.step_count = current_step
        if not feedback_batch:
            return

        for fb in feedback_batch:
            h = fb.horizon
            y_mean = float(np.mean(fb.base_samples))
            y_std = float(np.std(fb.base_samples)) + 1e-6
            residual = fb.y_true - y_mean
            realized_scale = abs(residual) / y_std

            # Age of feedback: arrival_step - origin_step = h
            age = fb.arrival_step - fb.origin_step
            staleness_weight = math.exp(-0.05 * age)

            # Store in memory buffer
            self.memory_buffer.append((residual, realized_scale, staleness_weight))
            self._update_prototypes(residual, realized_scale**2, current_step)

            # Update horizon local states
            self.h_loc[h] = (1.0 - self.lr * staleness_weight) * self.h_loc[h] + (
                self.lr * staleness_weight * residual
            )
            raw_scale = (1.0 - self.lr * staleness_weight) * self.h_scale[h] + (
                self.lr * staleness_weight * max(0.2, min(5.0, realized_scale))
            )
            self.h_scale[h] = max(0.1, min(5.0, raw_scale))

    def predict(self, horizon: int, base_samples: np.ndarray, current_step: int) -> np.ndarray:
        loc = self.h_loc.get(horizon, 0.0)
        scale = self.h_scale.get(horizon, 1.0)

        # Prototype blending if active
        if self.prototypes:
            weights = [math.exp(-0.02 * (current_step - p.last_seen)) for p in self.prototypes]
            total_w = sum(weights) + 1e-8
            proto_loc = sum(w * p.mean_residual for w, p in zip(weights, self.prototypes)) / total_w
            # Blend 70% horizon-specific, 30% prototype regime
            effective_loc = 0.7 * loc + 0.3 * proto_loc
        else:
            effective_loc = loc

        base_mean = np.mean(base_samples)
        adjusted = base_mean + effective_loc + scale * (base_samples - base_mean)
        return adjusted


# ==============================================================================
# 2. PADRE: Pattern-Aware Delayed Residual Estimation
# ==============================================================================

class PadreProbabilistic(BaseCompetitor):
    """
    PADRE: Pattern-Aware Delayed Residual Estimation.
    
    Decomposes delayed multi-horizon residuals into auto-regressive error
    signatures. Tracks horizon correlation lag matrices to project stale
    feedback forward to the current time origin.
    """

    def __init__(
        self,
        horizons: List[int],
        max_lag: int = 16,
        alpha_ema: float = 0.12,
        damping: float = 0.90,
    ):
        super().__init__("PADRE", horizons)
        self.max_lag = max_lag
        self.alpha_ema = alpha_ema
        self.damping = damping

        # Residual history ring buffer: step -> dict(horizon -> error)
        self.residual_history: Dict[int, Dict[int, float]] = {}
        # Correlation matrix between horizon errors: (h_i, h_j) -> corr
        self.horizon_corr: Dict[Tuple[int, int], float] = {
            (h1, h2): 1.0 if h1 == h2 else 0.5 for h1 in self.horizons for h2 in self.horizons
        }
        self.h_loc = {h: 0.0 for h in self.horizons}
        self.h_scale = {h: 1.0 for h in self.horizons}

    def reset(self) -> None:
        super().reset()
        self.residual_history.clear()
        self.h_loc = {h: 0.0 for h in self.horizons}
        self.h_scale = {h: 1.0 for h in self.horizons}
        self.horizon_corr = {
            (h1, h2): 1.0 if h1 == h2 else 0.5 for h1 in self.horizons for h2 in self.horizons
        }

    def update(self, feedback_batch: List[ForecastFeedback], current_step: int) -> None:
        self.step_count = current_step
        if not feedback_batch:
            return

        if current_step not in self.residual_history:
            self.residual_history[current_step] = {}

        for fb in feedback_batch:
            h = fb.horizon
            y_mean = float(np.mean(fb.base_samples))
            y_std = float(np.std(fb.base_samples)) + 1e-6
            err = fb.y_true - y_mean
            scale_ratio = abs(err) / y_std

            self.residual_history[current_step][h] = err

            # Update horizon corr with available partner horizons
            for other_h, other_err in self.residual_history[current_step].items():
                pair = (min(h, other_h), max(h, other_h))
                cur_corr = self.horizon_corr.get(pair, 0.5)
                # Incremental correlation proxy
                sign_match = 1.0 if (err * other_err > 0) else -0.5
                self.horizon_corr[pair] = 0.9 * cur_corr + 0.1 * sign_match

            # Direct state smoothing
            delay = fb.arrival_step - fb.origin_step
            forward_discount = self.damping ** delay
            self.h_loc[h] = (1.0 - self.alpha_ema) * self.h_loc[h] + self.alpha_ema * forward_discount * err
            raw_sc = (1.0 - self.alpha_ema) * self.h_scale[h] + self.alpha_ema * scale_ratio
            self.h_scale[h] = max(0.1, min(5.0, raw_sc))

        # Evict old history
        oldest_to_keep = current_step - self.max_lag
        keys_to_del = [k for k in self.residual_history if k < oldest_to_keep]
        for k in keys_to_del:
            del self.residual_history[k]

    def predict(self, horizon: int, base_samples: np.ndarray, current_step: int) -> np.ndarray:
        loc = self.h_loc.get(horizon, 0.0)
        scale = self.h_scale.get(horizon, 1.0)
        base_mean = np.mean(base_samples)
        return base_mean + loc + scale * (base_samples - base_mean)


# ==============================================================================
# 3. StayCALM: Calibrated Adaptive Online Aggregator with Overlapping Discount
# ==============================================================================

class StayCalmAggregator(BaseCompetitor):
    """
    StayCALM: Calibrated Adaptive Online Aggregator.
    
    Maintains an ensemble of horizon experts with multi-expert exponential
    weighting. Dis-counts overlapping loss updates using an effective sample
    size penalty inspired by conformal aggregation under non-stationary shifts.
    """

    def __init__(
        self,
        horizons: List[int],
        eta: float = 0.10,
        beta_overlap: float = 0.50,
    ):
        super().__init__("StayCALM", horizons)
        self.eta = eta
        self.beta_overlap = beta_overlap

        # Expert pool per horizon:
        # 0: Unadjusted baseline
        # 1: Fast EWMA tracker
        # 2: Slow EWMA tracker
        # 3: Scale-only tracker
        self.expert_names = ["ZeroShot", "FastEWMA", "SlowEWMA", "ScaleOnly"]
        self.n_experts = len(self.expert_names)

        # Expert weights per horizon: shape (n_experts,)
        self.weights: Dict[int, np.ndarray] = {
            h: np.ones(self.n_experts) / self.n_experts for h in self.horizons
        }
        # Expert parameter states
        self.fast_loc: Dict[int, float] = {h: 0.0 for h in self.horizons}
        self.slow_loc: Dict[int, float] = {h: 0.0 for h in self.horizons}
        self.scale_state: Dict[int, float] = {h: 1.0 for h in self.horizons}

    def reset(self) -> None:
        super().reset()
        self.weights = {h: np.ones(self.n_experts) / self.n_experts for h in self.horizons}
        self.fast_loc = {h: 0.0 for h in self.horizons}
        self.slow_loc = {h: 0.0 for h in self.horizons}
        self.scale_state = {h: 1.0 for h in self.horizons}

    def _expert_predict(self, expert_idx: int, horizon: int, base_samples: np.ndarray) -> np.ndarray:
        base_mean = np.mean(base_samples)
        centered = base_samples - base_mean

        if expert_idx == 0:  # ZeroShot
            return base_samples
        elif expert_idx == 1:  # FastEWMA
            return base_mean + self.fast_loc[horizon] + centered
        elif expert_idx == 2:  # SlowEWMA
            return base_mean + self.slow_loc[horizon] + centered
        elif expert_idx == 3:  # ScaleOnly
            return base_mean + self.scale_state[horizon] * centered
        return base_samples

    def update(self, feedback_batch: List[ForecastFeedback], current_step: int) -> None:
        self.step_count = current_step
        if not feedback_batch:
            return

        for fb in feedback_batch:
            h = fb.horizon
            y = fb.y_true
            base_m = float(np.mean(fb.base_samples))
            base_s = float(np.std(fb.base_samples)) + 1e-6
            err = y - base_m

            # Overlap discount factor
            overlap_penalty = 1.0 / (1.0 + self.beta_overlap * math.log2(max(1, h)))

            # Evaluate pinball / absolute loss for each expert on this arrival
            expert_losses = np.zeros(self.n_experts)
            for i in range(self.n_experts):
                pred_samples = self._expert_predict(i, h, fb.base_samples)
                pred_mean = np.mean(pred_samples)
                expert_losses[i] = abs(y - pred_mean)

            # Exponential loss update with overlap penalty
            scaled_loss = expert_losses * overlap_penalty
            self.weights[h] *= np.exp(-self.eta * (scaled_loss - np.min(scaled_loss)))
            self.weights[h] /= np.sum(self.weights[h])

            # Update underlying individual expert trackers
            self.fast_loc[h] = 0.80 * self.fast_loc[h] + 0.20 * err
            self.slow_loc[h] = 0.95 * self.slow_loc[h] + 0.05 * err
            sc = abs(err) / base_s
            self.scale_state[h] = 0.90 * self.scale_state[h] + 0.10 * max(0.1, min(5.0, sc))

    def predict(self, horizon: int, base_samples: np.ndarray, current_step: int) -> np.ndarray:
        w = self.weights[horizon]
        out = np.zeros_like(base_samples)
        for i in range(self.n_experts):
            out += w[i] * self._expert_predict(i, horizon, base_samples)
        return out


# ==============================================================================
# 4. AcMCP: Asymmetric Multi-Horizon Online Conformal Prediction
# ==============================================================================

class AcMcpAdapter(BaseCompetitor):
    """
    AcMCP: Asymmetric Multi-Horizon Online Conformal Prediction.
    
    Adapts coverage intervals dynamically using non-exchangeable conformal steps
    (Gibbs & Candes 2021 style) generalized to multi-horizon delayed feedback.
    Maintains separate lower and upper quantile score adjustments.
    """

    def __init__(
        self,
        horizons: List[int],
        target_coverage: float = 0.90,
        gamma_step: float = 0.05,
        alpha_ema: float = 0.08,
    ):
        super().__init__("AcMCP", horizons)
        self.target_coverage = target_coverage
        self.alpha_coverage = 1.0 - target_coverage
        self.gamma_step = gamma_step
        self.alpha_ema = alpha_ema

        # Horizon-specific conformal quantile adjustments (lower, upper)
        self.q_lower: Dict[int, float] = {h: 0.0 for h in self.horizons}
        self.q_upper: Dict[int, float] = {h: 0.0 for h in self.horizons}
        self.h_loc: Dict[int, float] = {h: 0.0 for h in self.horizons}

    def reset(self) -> None:
        super().reset()
        self.q_lower = {h: 0.0 for h in self.horizons}
        self.q_upper = {h: 0.0 for h in self.horizons}
        self.h_loc = {h: 0.0 for h in self.horizons}

    def update(self, feedback_batch: List[ForecastFeedback], current_step: int) -> None:
        self.step_count = current_step
        if not feedback_batch:
            return

        for fb in feedback_batch:
            h = fb.horizon
            y = fb.y_true
            base_m = float(np.mean(fb.base_samples))
            err = y - base_m

            # Track mean shift
            self.h_loc[h] = (1.0 - self.alpha_ema) * self.h_loc[h] + self.alpha_ema * err

            # Evaluate empirical coverage violation
            # Target coverage 1 - alpha: lower quantile alpha/2, upper 1 - alpha/2
            q_lo = float(np.quantile(fb.base_samples, self.alpha_coverage / 2.0)) - self.q_lower[h]
            q_hi = float(np.quantile(fb.base_samples, 1.0 - self.alpha_coverage / 2.0)) + self.q_upper[h]

            under_coverage = 1.0 if y < q_lo else 0.0
            over_coverage = 1.0 if y > q_hi else 0.0

            # Adaptive step sizes with delay damping
            step_effective = self.gamma_step / math.sqrt(h)

            # Lower bound step: expand if y < q_lo, contract otherwise
            grad_lo = under_coverage - (self.alpha_coverage / 2.0)
            self.q_lower[h] = max(0.0, self.q_lower[h] + step_effective * grad_lo)

            # Upper bound step: expand if y > q_hi, contract otherwise
            grad_hi = over_coverage - (self.alpha_coverage / 2.0)
            self.q_upper[h] = max(0.0, self.q_upper[h] + step_effective * grad_hi)

    def predict(self, horizon: int, base_samples: np.ndarray, current_step: int) -> np.ndarray:
        base_mean = np.mean(base_samples)
        centered = base_samples - base_mean
        loc = self.h_loc[horizon]

        # Stretch/shift quantiles according to conformal margins
        q_lo_val = self.q_lower[horizon]
        q_hi_val = self.q_upper[horizon]
        avg_expansion = 1.0 + (q_lo_val + q_hi_val) / (2.0 * (np.std(base_samples) + 1e-6))
        avg_expansion = max(0.2, min(4.0, avg_expansion))

        adjusted = base_mean + loc + avg_expansion * centered
        return adjusted


# ==============================================================================
# 5. RLS-TS: Multi-Horizon Recursive Least Squares with Horizon Covariance
# ==============================================================================

class RecursiveLeastSquaresAdapter(BaseCompetitor):
    """
    RLS-TS: Multi-Horizon Recursive Least Squares Adapter.
    
    Maintains a joint parameter vector across horizons with an inverse
    covariance matrix P. Handles delayed arrivals via rank-1 Sherman-Morrison
    updates with exponential forgetting factor lambda.
    """

    def __init__(
        self,
        horizons: List[int],
        forgetting_factor: float = 0.98,
        delta_regularizer: float = 1.0,
    ):
        super().__init__("RLS-TS", horizons)
        self.lam = forgetting_factor
        self.delta = delta_regularizer
        self.dim = len(self.horizons)
        self.h_to_idx = {h: i for i, h in enumerate(self.horizons)}

        # Weight vector theta in R^dim, Covariance P in R^(dim x dim)
        self.theta = np.zeros(self.dim)
        self.P = np.eye(self.dim) * (1.0 / self.delta)
        self.scale_tracker = {h: 1.0 for h in self.horizons}

    def reset(self) -> None:
        super().reset()
        self.theta = np.zeros(self.dim)
        self.P = np.eye(self.dim) * (1.0 / self.delta)
        self.scale_tracker = {h: 1.0 for h in self.horizons}

    def update(self, feedback_batch: List[ForecastFeedback], current_step: int) -> None:
        self.step_count = current_step
        if not feedback_batch:
            return

        for fb in feedback_batch:
            h = fb.horizon
            idx = self.h_to_idx[h]
            y = fb.y_true
            base_m = float(np.mean(fb.base_samples))
            base_s = float(np.std(fb.base_samples)) + 1e-6
            residual = y - base_m

            # Feature vector x: one-hot for horizon + shared component
            x = np.zeros(self.dim)
            x[idx] = 1.0
            # Weak coupling to adjacent horizons
            if idx > 0:
                x[idx - 1] = 0.25
            if idx < self.dim - 1:
                x[idx + 1] = 0.25

            # RLS gain vector: k = (P @ x) / (lambda + x^T @ P @ x)
            Px = self.P @ x
            denom = self.lam + float(x @ Px)
            k = Px / max(1e-6, denom)

            # Error using current weights
            pred_shift = float(self.theta @ x)
            e = residual - pred_shift

            # Weight update
            self.theta = self.theta + k * e

            # Sherman-Morrison covariance update: P = (P - k @ x^T @ P) / lambda
            self.P = (self.P - np.outer(k, Px)) / self.lam

            # Condition check on P
            if np.trace(self.P) > 1e4:
                self.P = np.eye(self.dim) * (1.0 / self.delta)

            # Scale tracking
            sc = abs(residual) / base_s
            self.scale_tracker[h] = 0.92 * self.scale_tracker[h] + 0.08 * max(0.2, min(5.0, sc))

    def predict(self, horizon: int, base_samples: np.ndarray, current_step: int) -> np.ndarray:
        idx = self.h_to_idx[horizon]
        x = np.zeros(self.dim)
        x[idx] = 1.0
        if idx > 0:
            x[idx - 1] = 0.25
        if idx < self.dim - 1:
            x[idx + 1] = 0.25

        loc_shift = float(self.theta @ x)
        scale = self.scale_tracker[horizon]

        base_mean = np.mean(base_samples)
        centered = base_samples - base_mean
        return base_mean + loc_shift + scale * centered


def get_all_competitors(horizons: List[int]) -> Dict[str, BaseCompetitor]:
    """Factory helper returning initialized instances of all 5 competitors."""
    return {
        "PROCEED": ProceedProbabilistic(horizons),
        "PADRE": PadreProbabilistic(horizons),
        "StayCALM": StayCalmAggregator(horizons),
        "AcMCP": AcMcpAdapter(horizons),
        "RLS-TS": RecursiveLeastSquaresAdapter(horizons),
    }
