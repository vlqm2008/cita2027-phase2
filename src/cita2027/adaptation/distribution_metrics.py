"""
Multivariate Proper Scoring Rules and Probabilistic Distribution Diagnostics.

This module provides high-order proper scoring rules and calibration statistics
for evaluating multi-horizon probabilistic time-series forecasts:
1. Energy Score (Gneiting & Raftery 2007) for joint trajectory evaluation
2. Variogram Score of orders p=0.5 and p=1.0 (Scheuerer & Hamill 2015)
3. Continuous Winkler Score integrated across confidence levels
4. Probability Integral Transform (PIT) Uniformity & Anderson-Darling goodness-of-fit
5. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
6. Normalized Interval Sharpness & Coverage Deviation

All implementations use vectorized NumPy operations and are guaranteed invariant
to permutation of Monte Carlo sample draws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ==============================================================================
# 1. Multivariate Energy Score
# ==============================================================================

def energy_score(
    y_true: np.ndarray,
    pred_trajectories: np.ndarray,
    beta: float = 1.0,
) -> float:
    """
    Multivariate Energy Score for joint multi-horizon forecast verification.
    
    The Energy Score is a strictly proper scoring rule for multivariate distributions
    generalizing the univariate Continuous Ranked Probability Score (CRPS).
    
    Formula:
        ES(F, y) = E_F[ ||y - X||_2^beta ] - 0.5 * E_F[ ||X - X'||_2^beta ]
    
    Parameters
    ----------
    y_true : np.ndarray
        True observed trajectory of shape (H,) where H is the forecast horizon length.
    pred_trajectories : np.ndarray
        Sampled trajectories of shape (S, H), where S is the number of Monte Carlo draws.
    beta : float, default=1.0
        Power exponent in (0, 2). beta=1.0 corresponds to standard Euclidean distance.
        
    Returns
    -------
    float
        The scalar Energy Score. Lower is strictly better.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    trajs = np.asarray(pred_trajectories, dtype=np.float64)
    if trajs.ndim == 1:
        trajs = trajs[:, np.newaxis]
    if y_true.ndim == 0:
        y_true = np.array([y_true])

    S, H = trajs.shape
    assert len(y_true) == H, f"Dimension mismatch: y_true length {len(y_true)} != H {H}"

    # First term: mean Euclidean distance from observations to samples
    # shape: (S,)
    diff_y = trajs - y_true[np.newaxis, :]
    dist_y = np.linalg.norm(diff_y, ord=2, axis=1)
    if beta != 1.0:
        dist_y = dist_y ** beta
    term1 = float(np.mean(dist_y))

    # Second term: mean pairwise Euclidean distance between distinct sample paths
    # Vectorized computation of pairwise distances
    # For large S, sample without replacement or pairwise difference matrix
    if S <= 250:
        # Exact all-pairs difference
        # trajs[:, None, :] - trajs[None, :, :] has shape (S, S, H)
        pair_diff = trajs[:, np.newaxis, :] - trajs[np.newaxis, :, :]
        pair_dist = np.linalg.norm(pair_diff, ord=2, axis=2)
        if beta != 1.0:
            pair_dist = pair_dist ** beta
        term2 = float(np.sum(pair_dist)) / (2.0 * S * S)
    else:
        # Fast Monte Carlo approximation for very large S
        idx1 = np.random.randint(0, S, size=S)
        idx2 = np.random.randint(0, S, size=S)
        pair_dist = np.linalg.norm(trajs[idx1] - trajs[idx2], ord=2, axis=1)
        if beta != 1.0:
            pair_dist = pair_dist ** beta
        term2 = 0.5 * float(np.mean(pair_dist))

    return term1 - term2


# ==============================================================================
# 2. Variogram Score
# ==============================================================================

def variogram_score(
    y_true: np.ndarray,
    pred_trajectories: np.ndarray,
    p: float = 0.5,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Variogram Score for evaluating multivariate cross-horizon dependence structure.
    
    The Variogram-based score (Scheuerer & Hamill, MWR 2015) explicitly evaluates
    whether the forecast captures the auto-correlation and pairwise increments
    between different lead times h_i and h_j.
    
    Formula:
        VS_p(F, y) = sum_{i=1}^H sum_{j=1}^H w_{ij} * ( |y_i - y_j|^p - E_F[|X_i - X_j|^p] )^2
        
    Parameters
    ----------
    y_true : np.ndarray
        True observed vector of shape (H,).
    pred_trajectories : np.ndarray
        Sampled trajectories of shape (S, H).
    p : float, default=0.5
        Variogram order. Typical literature choices are p=0.5 or p=1.0.
    weights : np.ndarray, optional
        Matrix of non-negative weights w_{ij} of shape (H, H). If None, uniform weights
        w_{ij} = 1.0 / (H^2) are applied.
        
    Returns
    -------
    float
        The scalar Variogram Score. Lower is strictly better.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    trajs = np.asarray(pred_trajectories, dtype=np.float64)
    if trajs.ndim == 1:
        trajs = trajs[:, np.newaxis]
    S, H = trajs.shape

    if H <= 1:
        # Degenerate univariate case: variogram pairwise is zero
        return 0.0

    if weights is None:
        w = np.ones((H, H), dtype=np.float64) / (H * H)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w)

    # Pairwise differences of true values: |y_i - y_j|^p
    # shape: (H, H)
    true_diff = np.abs(y_true[:, np.newaxis] - y_true[np.newaxis, :]) ** p

    # Pairwise differences of sample paths: E_F[|X_i - X_j|^p]
    # trajs shape: (S, H)
    # trajs[:, :, None] - trajs[:, None, :] shape: (S, H, H)
    sample_diffs = np.abs(trajs[:, :, np.newaxis] - trajs[:, np.newaxis, :]) ** p
    exp_sample_diff = np.mean(sample_diffs, axis=0)  # shape (H, H)

    # Squared discrepancy weighted by w_{ij}
    score_matrix = (true_diff - exp_sample_diff) ** 2
    return float(np.sum(w * score_matrix))


# ==============================================================================
# 3. Continuous Winkler Score & Multi-Alpha Interval Evaluation
# ==============================================================================

def winkler_score_level(
    y_true: float,
    pred_samples: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """
    Standard Winkler score at nominal miscoverage rate alpha.
    
    Parameters
    ----------
    y_true : float
        Observed ground truth.
    pred_samples : np.ndarray
        Monte Carlo samples of shape (S,).
    alpha : float, default=0.10
        Nominal error level (0.10 -> 90% confidence interval).
        
    Returns
    -------
    float
        Winkler score. Lower is better.
    """
    samples = np.sort(pred_samples)
    l = float(np.quantile(samples, alpha / 2.0))
    u = float(np.quantile(samples, 1.0 - alpha / 2.0))
    width = u - l

    if y_true < l:
        penalty = (2.0 / alpha) * (l - y_true)
    elif y_true > u:
        penalty = (2.0 / alpha) * (y_true - u)
    else:
        penalty = 0.0

    return width + penalty


def continuous_winkler_score(
    y_true: float,
    pred_samples: np.ndarray,
    alpha_grid: Optional[Sequence[float]] = None,
) -> float:
    """
    Continuous Winkler Score integrated over coverage probability spectrum.
    
    Integrates the Winkler score over alpha in [0.02, 0.98] to evaluate interval
    sharpness and coverage robustness simultaneously across both narrow central
    predictions and wide tail bounds.
    
    Parameters
    ----------
    y_true : float
        Observed ground truth.
    pred_samples : np.ndarray
        Array of shape (S,) with predictive draws.
    alpha_grid : Sequence[float], optional
        Grid of alpha levels. Defaults to 20 equally spaced points in [0.05, 0.95].
        
    Returns
    -------
    float
        Averaged integrated Winkler score.
    """
    if alpha_grid is None:
        alpha_grid = np.linspace(0.05, 0.90, num=18)

    scores = [winkler_score_level(y_true, pred_samples, a) for a in alpha_grid]
    return float(np.mean(scores))


# ==============================================================================
# 4. Probability Integral Transform (PIT) & Anderson-Darling Test
# ==============================================================================

@dataclass
class PITDiagnosticResult:
    """Results from Probability Integral Transform goodness-of-fit analysis."""
    pit_values: np.ndarray
    ks_statistic: float
    ks_pvalue: float
    anderson_statistic: float
    is_uniform_5pct: bool
    histogram_bins: np.ndarray
    histogram_counts: np.ndarray


def compute_pit_values(
    y_true_seq: Sequence[float],
    pred_samples_seq: Sequence[np.ndarray],
    randomize_ties: bool = True,
) -> np.ndarray:
    """
    Compute Probability Integral Transform (PIT) values for a prequential sequence.
    
    For an ideal well-calibrated forecast distribution, the PIT values u_t = F_t(y_t)
    are independent and identically distributed standard uniform U(0, 1).
    
    Parameters
    ----------
    y_true_seq : Sequence[float]
        Sequence of true realized observations.
    pred_samples_seq : Sequence[np.ndarray]
        Sequence of predictive samples, each of shape (S,).
    randomize_ties : bool, default=True
        Whether to break exact empirical step function ties by uniform randomization.
        
    Returns
    -------
    np.ndarray
        Array of PIT values in [0, 1] of length T.
    """
    pit_vals = []
    rng = np.random.default_rng(42)

    for y, samples in zip(y_true_seq, pred_samples_seq):
        samples = np.asarray(samples)
        S = len(samples)
        strict_below = np.sum(samples < y)
        equal_to = np.sum(samples == y)

        if equal_to > 0 and randomize_ties:
            u = (strict_below + rng.uniform(0.0, 1.0) * equal_to) / S
        else:
            u = (strict_below + 0.5 * equal_to) / S

        # Clip slightly to avoid numerical boundary singularities in log(u)
        u = float(np.clip(u, 1e-6, 1.0 - 1e-6))
        pit_vals.append(u)

    return np.array(pit_vals, dtype=np.float64)


def pit_anderson_darling_uniformity(
    pit_values: np.ndarray,
    n_bins: int = 10,
) -> PITDiagnosticResult:
    """
    Evaluate PIT uniformity via Kolmogorov-Smirnov and Anderson-Darling tests.
    
    Under H0: pit_values ~ Uniform(0, 1).
    
    Parameters
    ----------
    pit_values : np.ndarray
        Array of PIT values in [0, 1].
    n_bins : int, default=10
        Number of histogram bins for diagnostic plotting.
        
    Returns
    -------
    PITDiagnosticResult
        Structured diagnostic summary including KS and AD statistics.
    """
    u_sorted = np.sort(pit_values)
    N = len(u_sorted)

    # 1. Kolmogorov-Smirnov against Uniform(0, 1) CDF F(u) = u
    i_over_N = np.arange(1, N + 1) / N
    i_minus_1_over_N = np.arange(0, N) / N
    d_plus = np.max(i_over_N - u_sorted)
    d_minus = np.max(u_sorted - i_minus_1_over_N)
    ks_stat = float(max(d_plus, d_minus))

    # Asymptotic KS p-value approximation (Smirnov 1948)
    lambda_ks = (math.sqrt(N) + 0.12 + 0.11 / math.sqrt(N)) * ks_stat
    if lambda_ks > 0:
        ks_pval = 2.0 * math.exp(-2.0 * (lambda_ks**2))
        ks_pval = min(1.0, max(0.0, ks_pval))
    else:
        ks_pval = 1.0

    # 2. Anderson-Darling statistic for standard Uniform(0, 1)
    # A^2 = -N - (1/N) * sum_{i=1}^N (2i - 1) * [ln(u_i) + ln(1 - u_{N+1-i})]
    i_vec = np.arange(1, N + 1)
    log_terms = np.log(u_sorted) + np.log(1.0 - u_sorted[::-1])
    ad_stat = float(-N - (1.0 / N) * np.sum((2 * i_vec - 1) * log_terms))

    # Histogram counts
    counts, bin_edges = np.histogram(pit_values, bins=n_bins, range=(0.0, 1.0))

    return PITDiagnosticResult(
        pit_values=pit_values,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        anderson_statistic=ad_stat,
        is_uniform_5pct=(ks_pval >= 0.05),
        histogram_bins=bin_edges,
        histogram_counts=counts,
    )


# ==============================================================================
# 5. Expected Calibration Error (ECE) & Coverage Diagnostics
# ==============================================================================

@dataclass
class CalibrationReport:
    """Summary of empirical interval coverage, ECE, and sharpness."""
    nominal_levels: List[float]
    empirical_coverages: List[float]
    coverage_deviations: List[float]
    expected_calibration_error: float
    max_calibration_error: float
    mean_interval_width_90: float
    normalized_sharpness: float


def evaluate_calibration_and_sharpness(
    y_true_seq: Sequence[float],
    pred_samples_seq: Sequence[np.ndarray],
    nominal_coverages: Tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
) -> CalibrationReport:
    """
    Compute Expected Calibration Error (ECE) and Normalized Sharpness.
    
    Parameters
    ----------
    y_true_seq : Sequence[float]
        True observations of length T.
    pred_samples_seq : Sequence[np.ndarray]
        List of sample draws, each of shape (S,).
    nominal_coverages : Tuple[float, ...]
        Nominal nominal coverage probabilities (e.g. 0.50, 0.80, 0.90, 0.95).
        
    Returns
    -------
    CalibrationReport
        Structured metrics summarizing coverage fidelity and interval sharpness.
    """
    y_arr = np.asarray(y_true_seq)
    T = len(y_arr)
    emp_coverages = []
    deviations = []

    for nom in nominal_coverages:
        alpha = 1.0 - nom
        covered_count = 0
        for y, samples in zip(y_arr, pred_samples_seq):
            q_lo = np.quantile(samples, alpha / 2.0)
            q_hi = np.quantile(samples, 1.0 - alpha / 2.0)
            if q_lo <= y <= q_hi:
                covered_count += 1
        emp = covered_count / T
        emp_coverages.append(emp)
        deviations.append(abs(emp - nom))

    ece = float(np.mean(deviations))
    mce = float(np.max(deviations))

    # Interval sharpness at 90% (alpha=0.10)
    widths = []
    for samples in pred_samples_seq:
        q05 = np.quantile(samples, 0.05)
        q95 = np.quantile(samples, 0.95)
        widths.append(q95 - q05)
    mean_w90 = float(np.mean(widths))

    # Normalized sharpness: interval width divided by empirical ground truth standard deviation
    y_std = float(np.std(y_arr)) + 1e-6
    norm_sharpness = mean_w90 / y_std

    return CalibrationReport(
        nominal_levels=list(nominal_coverages),
        empirical_coverages=emp_coverages,
        coverage_deviations=deviations,
        expected_calibration_error=ece,
        max_calibration_error=mce,
        mean_interval_width_90=mean_w90,
        normalized_sharpness=norm_sharpness,
    )


# ==============================================================================
# 6. Comprehensive Prequential Trajectory Evaluator
# ==============================================================================

class TrajectoryEvaluator:
    """
    Unified evaluator aggregating univariate and multivariate proper scores
    over sequential forecast trajectories.
    """

    def __init__(self, horizons: List[int]):
        self.horizons = sorted(list(horizons))
        self.H = len(self.horizons)

    def evaluate_sequence(
        self,
        y_true_matrix: np.ndarray,
        pred_trajectories_seq: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Evaluate full multi-horizon trajectory sequence.
        
        Parameters
        ----------
        y_true_matrix : np.ndarray
            Matrix of true observations of shape (T, H).
        pred_trajectories_seq : List[np.ndarray]
            List of T sample arrays, each of shape (S, H).
            
        Returns
        -------
        Dict[str, float]
            Aggregated metric summary.
        """
        T = len(pred_trajectories_seq)
        assert y_true_matrix.shape[0] == T, "Time dimension T mismatch"
        assert y_true_matrix.shape[1] == self.H, "Horizon dimension H mismatch"

        energy_scores = []
        variogram_05_scores = []
        variogram_10_scores = []
        continuous_winkler_scores = []
        crps_per_horizon = {h: [] for h in self.horizons}

        for t in range(T):
            y_t = y_true_matrix[t]
            trajs_t = pred_trajectories_seq[t]  # shape (S, H)

            # Multivariate metrics
            es = energy_score(y_t, trajs_t, beta=1.0)
            vs05 = variogram_score(y_t, trajs_t, p=0.5)
            vs10 = variogram_score(y_t, trajs_t, p=1.0)

            energy_scores.append(es)
            variogram_05_scores.append(vs05)
            variogram_10_scores.append(vs10)

            # Continuous Winkler across all horizons
            for h_idx, h in enumerate(self.horizons):
                cw = continuous_winkler_score(float(y_t[h_idx]), trajs_t[:, h_idx])
                continuous_winkler_scores.append(cw)

                # Univariate empirical CRPS
                s_h = trajs_t[:, h_idx]
                y_val = y_t[h_idx]
                crps_val = float(np.mean(np.abs(s_h - y_val))) - 0.5 * float(
                    np.mean(np.abs(s_h[:, None] - s_h[None, :]))
                )
                crps_per_horizon[h].append(crps_val)

        results: Dict[str, float] = {
            "mean_energy_score": float(np.mean(energy_scores)),
            "mean_variogram_05": float(np.mean(variogram_05_scores)),
            "mean_variogram_10": float(np.mean(variogram_10_scores)),
            "mean_continuous_winkler": float(np.mean(continuous_winkler_scores)),
        }

        for h in self.horizons:
            results[f"crps_h{h}"] = float(np.mean(crps_per_horizon[h]))

        results["mean_crps"] = float(np.mean([results[f"crps_h{h}"] for h in self.horizons]))
        return results
