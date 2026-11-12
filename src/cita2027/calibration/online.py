"""TASK-005A: causal, CPU-only online CQR calibration.

All returns and interval endpoints are in decimal daily-log-return units.
The controller never receives the current realized target until AFTER prediction.
No exchangeability or finite-sample coverage claim is made for time series.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil, isfinite, sqrt

import numpy as np

SCALES = ("none", "residual_ewma", "return_ewma")
CONTROLLERS = ("rolling_cqr", "fixed_gamma_aci", "variable_gamma_aci")


@dataclass(frozen=True)
class Parameters:
    alpha: float = 0.20  # q10/q90 native interval targets nominal 80% coverage.
    window: int = 252
    decay: float = 0.94
    gamma: float = 0.01
    min_scale: float = 1e-6  # decimal log return
    driver_min: float = 0.5
    driver_max: float = 2.0
    correction: str = "expansion_only"  # signed_interval is a separate ablation

    def __post_init__(self):
        if not (0 < self.alpha < 1 and self.window >= 2 and 0 < self.decay < 1
                and self.gamma >= 0 and self.min_scale > 0
                and 0 < self.driver_min <= 1 <= self.driver_max
                and self.correction in ("expansion_only", "signed_interval")):
            raise ValueError("Invalid calibration specification")


def signed_score(y: float, lower: float, upper: float) -> float:
    """CQR score can be negative when the realized value is inside the raw interval."""
    if not all(isfinite(x) for x in (y, lower, upper)) or lower > upper:
        raise ValueError("Invalid observed value or forecast interval")
    return max(lower - y, y - upper)


def conformal_order_statistic(scores, alpha: float) -> float:
    """k=ceil((n+1)(1-alpha)); 1-indexed k-th smallest, +inf when k=n+1.

    Ties remain tied (no randomization). Exact equality to an endpoint is covered.
    This is NOT numpy.quantile(..., method='linear').
    """
    a = np.asarray(list(scores), dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or not 0 < alpha < 1:
        raise ValueError("Nonempty finite score list and alpha in (0,1) required")
    k = ceil((len(a) + 1) * (1 - alpha))
    return float(np.partition(a, k - 1)[k - 1]) if k <= len(a) else float("inf")


def interval(lower: float, upper: float, threshold: float, scale: float,
             correction: str = "expansion_only") -> tuple[float, float]:
    if not (all(isfinite(x) for x in (lower, upper, scale)) and lower <= upper and scale > 0):
        raise ValueError("Invalid interval or positive scale")
    if correction not in ("expansion_only", "signed_interval"):
        raise ValueError("Unknown correction convention")
    if correction == "expansion_only":
        threshold = max(0.0, threshold)
    adjust = threshold * scale
    a, b = lower - adjust, upper + adjust
    if not all(isfinite(x) for x in (a, b)):
        raise ValueError("Unbounded interval; report separately, do not score as finite")
    if a > b:
        raise ValueError("Signed-CQR empty set; cannot evaluate with ordinary Winkler score")
    return a, b


def interval_metrics(rows: list[dict], alpha: float, rolling_window: int = 50) -> dict:
    if not rows:
        raise ValueError("No evaluation rows")
    low = np.array([r["lower"] for r in rows], dtype=float)
    up = np.array([r["upper"] for r in rows], dtype=float)
    y = np.array([r["target_log_return"] for r in rows], dtype=float)
    if not (np.isfinite(low).all() and np.isfinite(up).all() and np.isfinite(y).all()
            and np.all(low <= up)):
        raise ValueError("Invalid calibrated endpoints")
    low_miss, high_miss = y < low, y > up
    width = up - low
    lower_penalty = (2 / alpha) * (low - y) * low_miss
    upper_penalty = (2 / alpha) * (y - up) * high_miss
    winkler = width + lower_penalty + upper_penalty
    miss = low_miss | high_miss
    n = len(rows)
    rolling = [float(np.mean(~miss[j - rolling_window + 1:j + 1]))
               for j in range(rolling_window - 1, n)]
    return {
        "n": n, "winkler_mean": float(np.mean(winkler)),
        "width_mean": float(np.mean(width)),
        "width_component_mean": float(np.mean(width)),
        "lower_penalty_mean": float(np.mean(lower_penalty)),
        "upper_penalty_mean": float(np.mean(upper_penalty)),
        "coverage": float(np.mean(~miss)),
        "lower_miss_rate": float(np.mean(low_miss)),
        "upper_miss_rate": float(np.mean(high_miss)),
        "rolling_empirical_coverage_min": min(rolling) if rolling else None,
        "rolling_empirical_coverage_max": max(rolling) if rolling else None,
        "rolling_window": rolling_window,
        "conditional_coverage": None,  # Not identifiable/validated from a rolling statistic.
    }


def calibrate(records: list[dict], initial_context: np.ndarray, params: Parameters,
              scale_kind: str, controller: str, limit: int | None = None) -> list[dict]:
    """Chronological predict -> observe -> update, with strictly prior rolling scores.

    initial_context is the *first origin's* 256 past/present returns, no future.
    ACI is initialized at nominal alpha at first evaluated row, NOT during burn-in.
    Both EWMA states start from the same causal RMS initial context. At origin i,
    vol_ret_i and vol_res_i contain observations through i only. After forecasting
    and observing target at i+1, they are advanced to the next origin.
    """
    if scale_kind not in SCALES or controller not in CONTROLLERS:
        raise ValueError("Unknown factorial cell")
    if limit is not None:
        records = records[:limit]
    if len(records) <= params.window:
        raise ValueError("Need strictly more records than calibration burn-in")
    first = np.asarray(initial_context, dtype=float)
    if first.shape != (256,) or not np.isfinite(first).all():
        raise ValueError("Initial causal context must be 256 finite returns")
    base_var = max(float(np.mean(first**2)), params.min_scale**2)
    var_ret, var_res = base_var, base_var
    reference = max(sqrt(base_var), params.min_scale)
    scores: deque[float] = deque(maxlen=params.window)
    alpha_t = params.alpha
    output = []
    for j, rec in enumerate(records):
        i = rec["origin_index"]
        if (j and (i != records[j - 1]["origin_index"] + 1 or
                   rec["target_index"] != i + 1)):
            raise ValueError("Forecast stream must be contiguous in trading-row index")
        if rec["target_index"] != i + 1 or rec["seed"] != 2027 + i:
            raise ValueError("Causal index/seed violation")
        lo, mid, hi, y = (float(rec[k]) for k in ("q10", "q50", "q90", "target_log_return"))
        if not (np.isfinite([lo, mid, hi, y]).all() and lo <= mid <= hi):
            raise ValueError("Invalid raw quantiles or target")
        ret_sigma = max(sqrt(var_ret), params.min_scale)
        res_sigma = max(sqrt(var_res), params.min_scale)
        scale = {"none": 1.0, "residual_ewma": res_sigma, "return_ewma": ret_sigma}[scale_kind]
        # Common return-based gamma driver across all three scale conditions.
        driver = float(np.clip(ret_sigma / reference, params.driver_min, params.driver_max))
        gamma_t = params.gamma * (driver if controller == "variable_gamma_aci" else 1.0)
        if j >= params.window:
            # Fixed window n is known. Dynamic alpha never enters calibration buffer.
            active_alpha = params.alpha if controller == "rolling_cqr" else alpha_t
            q = conformal_order_statistic(scores, active_alpha)
            low, up = interval(lo, hi, q, scale, params.correction)
            miss = int(y < low or y > up)
            output.append({
                "origin_index": i, "target_index": i + 1,
                "origin_date": rec["origin_date"], "target_date": rec["target_date"],
                "target_log_return": y, "raw_q10": lo, "raw_q50": mid,
                "raw_q90": hi, "lower": low, "upper": up,
                "scale": scale, "return_volatility": ret_sigma,
                "residual_volatility": res_sigma, "gamma_driver": driver,
                "gamma_t": gamma_t if controller != "rolling_cqr" else 0.0,
                "alpha_t": active_alpha, "score_threshold": q,
                "miss": miss, "burn_in": False,
            })
            if controller != "rolling_cqr":
                # Nonstandard bounded ACI: ensures finite order statistic at n=W.
                floor = 1.0 / (params.window + 1)
                alpha_t = float(np.clip(alpha_t + gamma_t * (params.alpha - miss),
                                        floor, 1.0 - floor))
        # Feedback is visible ONLY after prediction. Score uses origin-time scale.
        scores.append(signed_score(y, lo, hi) / scale)
        var_res = params.decay * var_res + (1 - params.decay) * (y - mid)**2
        var_ret = params.decay * var_ret + (1 - params.decay) * y**2
    return output
