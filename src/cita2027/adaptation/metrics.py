"""Distributional and interval evaluation metrics for probabilistic forecasting."""
from __future__ import annotations

import numpy as np


def crps_empirical_samples(samples: np.ndarray, y: float) -> float:
    """Exact computation of Continuous Ranked Probability Score from sample array.

    Formula:
      CRPS(F, y) = E[|X - y|] - 0.5 * E[|X - X'|]
      computed efficiently via sorting in O(M log M).
    """
    s = np.sort(np.asarray(samples, dtype=np.float64))
    m = len(s)
    if m < 2:
        raise ValueError("CRPS requires at least 2 samples.")

    # E[|X - y|]
    term1 = np.mean(np.abs(s - y))

    # 0.5 * E[|X - X'|] using sorted order statistic identity:
    # sum_{i,j} |s_i - s_j| = 2 * sum_{k=1}^M (2k - M - 1) * s_k
    # => 0.5 * (1 / M^2) * sum_{i,j} |s_i - s_j| = (1 / M^2) * sum_{k=1}^M (2k - M - 1) * s_k
    weights = 2 * np.arange(1, m + 1) - m - 1
    term2 = np.sum(weights * s) / (m * m)

    val = float(term1 - term2)
    return max(0.0, val)


def winkler_score(lower: float, upper: float, y: float, alpha: float = 0.20) -> float:
    """Winkler score for (1 - alpha) prediction interval [lower, upper]."""
    if upper < lower:
        raise ValueError(f"Invalid interval: lower ({lower}) > upper ({upper})")
    width = upper - lower
    score = width
    if y < lower:
        score += (2.0 / alpha) * (lower - y)
    elif y > upper:
        score += (2.0 / alpha) * (y - upper)
    return float(score)


def interval_metrics(samples: np.ndarray, y: float, alpha: float = 0.20) -> dict[str, float]:
    """Compute empirical coverage, width, winkler score at nominal 1 - alpha."""
    s = np.asarray(samples, dtype=np.float64)
    q_low = float(np.quantile(s, alpha / 2.0))
    q_high = float(np.quantile(s, 1.0 - alpha / 2.0))
    q_med = float(np.median(s))

    w = q_high - q_low
    covered = 1.0 if (q_low <= y <= q_high) else 0.0
    ws = winkler_score(q_low, q_high, y, alpha=alpha)
    mae = float(abs(y - q_med))
    crps = crps_empirical_samples(s, y)

    return {
        "crps": crps,
        "mae": mae,
        "winkler": ws,
        "width": w,
        "coverage": covered,
        "q_low": q_low,
        "q_med": q_med,
        "q_high": q_high,
    }
