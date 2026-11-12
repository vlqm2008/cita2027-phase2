"""Statistical significance and hypothesis testing for adaptation experiments.

Implements Circular Block Bootstrap (Politis & Romano 1994), Wilcoxon signed-rank tests,
and Holm-Bonferroni family-wise error rate corrections.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats


def circular_block_bootstrap_ci(
    diff: np.ndarray,
    block_lengths: List[int] = [5, 10, 20],
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 2027,
) -> Dict[str, Any]:
    """Compute circular block bootstrap confidence intervals for paired differences.

    Args:
        diff: 1D array of differences (e.g. loss_candidate - loss_baseline)
        block_lengths: list of block sizes b
        num_bootstrap: number of resamples
        confidence_level: target coverage (0.95 for 95% CI)
        seed: random seed

    Returns:
        Dict with mean delta, per-block CI bounds, and whether all exclude zero.
    """
    diff = np.asarray(diff, dtype=np.float64)
    n = len(diff)
    if n == 0:
        raise ValueError("diff array cannot be empty")

    mean_diff = float(np.mean(diff))
    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    rng = np.random.RandomState(seed)
    # Circular wrap buffer
    extended = np.concatenate([diff, diff])

    ci_results = {}
    all_exclude_zero = True

    for b in block_lengths:
        b = min(b, n)
        n_blocks = int(np.ceil(n / b))
        boot_means = np.empty(num_bootstrap, dtype=np.float64)

        for i in range(num_bootstrap):
            start_indices = rng.randint(0, n, size=n_blocks)
            sample_blocks = [extended[idx:idx + b] for idx in start_indices]
            resample = np.concatenate(sample_blocks)[:n]
            boot_means[i] = np.mean(resample)

        lo = float(np.percentile(boot_means, lower_pct))
        hi = float(np.percentile(boot_means, upper_pct))
        ci_results[f"b{b}"] = (lo, hi)

        if lo <= 0.0 <= hi:
            all_exclude_zero = False

    return {
        "n": n,
        "mean_diff": mean_diff,
        "ci": ci_results,
        "all_exclude_zero": all_exclude_zero,
    }


def paired_wilcoxon_test(
    candidate_losses: np.ndarray,
    reference_losses: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Compute paired Wilcoxon signed-rank test on loss differences.

    Returns:
        (statistic, p_value)
    """
    candidate_losses = np.asarray(candidate_losses, dtype=np.float64)
    reference_losses = np.asarray(reference_losses, dtype=np.float64)
    diff = candidate_losses - reference_losses

    # Exclude exact zeros for Wilcoxon
    nonzero_diff = diff[diff != 0]
    if len(nonzero_diff) < 10:
        return 0.0, 1.0

    try:
        res = stats.wilcoxon(nonzero_diff, alternative=alternative)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Apply Holm-Bonferroni step-down procedure.

    Returns:
        (adjusted_p_values, rejected_hypotheses)
    """
    m = len(p_values)
    if m == 0:
        return [], []

    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adj_p = [0.0] * m
    rejected = [False] * m

    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p):
        k = rank + 1
        adj = (m - k + 1) * p
        adj = min(adj, 1.0)
        cum_max = max(cum_max, adj)
        adj_p[orig_idx] = cum_max

    for i in range(m):
        rejected[i] = adj_p[i] < alpha

    return adj_p, rejected
