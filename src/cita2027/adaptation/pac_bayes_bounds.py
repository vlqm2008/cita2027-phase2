"""PAC-Bayesian and Information-Theoretic Generalization Bounds for Cross-Horizon Adaptation.

This module formalizes Theorem 2:
Theorem 2 (Cross-Horizon PAC-Bayesian Generalization Bound for Asynchronous Feedback).
Let H = {h_1, ..., h_K} be the set of forecast horizons with maximum feedback delay tau_max.
Under non-stationary prequential observations where conditional distributions drift with
total variation budget V_T = sum_{t=1}^T TV(P_t, P_{t-1}), for any prior pi_0 and
adapted posterior rho_T over the shared representation manifold, with probability at least
1 - delta over the causal filtration:

    (1/T) sum_{t=1}^T CRPS(F_{t,h}^{adapted}, y_{t+h})
      <= (1/T) sum_{t=1}^T CRPS(F_{t,h}^*, y_{t+h})
         + sqrt( (KL(rho_T || pi_0) + log(2*sqrt(T)/delta)) / (2*T) )
         + C_h * (tau_h * V_T) / T
         + lambda_max(C_overlap) / (2 * sqrt(T))

where:
- KL(rho_T || pi_0) measures representation drift on the latent manifold.
- C_h is the L1-Lipschitz constant of the CRPS scoring rule with respect to location shift.
- tau_h is the horizon delay (tau_h = h).
- C_overlap is the multi-horizon overlap covariance matrix.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PACBayesBoundResult:
    """Container for Theorem 2 theoretical evaluation."""
    horizon: int
    time_horizon_T: int
    empirical_risk: float
    theoretical_upper_bound: float
    kl_divergence: float
    complexity_penalty: float
    drift_penalty: float
    overlap_penalty: float
    bound_slack: float  # theoretical_bound - empirical_risk (must be >= 0)
    bound_holds: bool


class CrossHorizonPACBayesEvaluator:
    """Computes and validates Theorem 2 PAC-Bayesian generalization bounds."""

    def __init__(
        self,
        horizons: List[int],
        latent_dim: int = 2,
        delta: float = 0.05,
        crps_lipschitz_c: float = 1.0,
        prior_variance: float = 1.0,
    ) -> None:
        self.horizons = sorted(horizons)
        self.latent_dim = latent_dim
        self.delta = delta
        self.crps_lipschitz_c = crps_lipschitz_c
        self.prior_variance = prior_variance

    def compute_manifold_kl_divergence(
        self,
        shared_state_history: np.ndarray,
    ) -> float:
        """Computes Gaussian KL divergence between empirical posterior and isotropic prior pi_0.

        pi_0 ~ N(0, sigma_0^2 * I)
        rho_T ~ N(mu_T, Sigma_T)
        """
        states = np.asarray(shared_state_history, dtype=np.float64)
        if states.ndim == 1:
            states = states.reshape(-1, 1)

        T, d = states.shape
        if T < 2:
            return 0.01

        mu_T = np.mean(states, axis=0)
        cov_T = np.cov(states, rowvar=False)
        if d == 1:
            cov_T = np.array([[float(cov_T)]])
        else:
            # Regularize covariance to ensure positive definiteness
            cov_T = cov_T + 1e-4 * np.eye(d)

        sigma_0_sq = self.prior_variance
        inv_sigma_0 = (1.0 / sigma_0_sq) * np.eye(d)

        # Tr(Sigma_0^{-1} Sigma_T)
        trace_term = np.trace(inv_sigma_0 @ cov_T)
        # (mu_0 - mu_T)^T Sigma_0^{-1} (mu_0 - mu_T)
        quad_term = float(mu_T.T @ inv_sigma_0 @ mu_T)
        # log( det(Sigma_0) / det(Sigma_T) )
        sign, logdet_T = np.linalg.slogdet(cov_T)
        logdet_0 = d * math.log(sigma_0_sq)
        logdet_term = logdet_0 - logdet_T

        kl = 0.5 * (trace_term + quad_term - d + logdet_term)
        return float(max(kl, 1e-5))

    def estimate_total_variation_drift(
        self,
        observations: np.ndarray,
        window: int = 20,
    ) -> float:
        """Estimates total variation path budget V_T via sliding-window Wasserstein-1 proxy."""
        obs = np.asarray(observations, dtype=np.float64)
        T = len(obs)
        if T <= window * 2:
            return 0.1

        tv_sum = 0.0
        for t in range(window, T - window, window):
            w1 = obs[t - window : t]
            w2 = obs[t : t + window]
            # Absolute median shift + scale change as TV lower bound
            med_diff = abs(float(np.median(w2) - np.median(w1)))
            scale_diff = abs(float(np.std(w2) - np.std(w1)))
            tv_sum += (med_diff + scale_diff)

        return float(tv_sum)

    def compute_overlap_spectral_norm(
        self,
        horizon: int,
        feedback_latencies: Dict[int, int],
    ) -> float:
        """Computes maximum eigenvalue of multi-horizon overlap correlation matrix."""
        K = len(self.horizons)
        overlap_matrix = np.zeros((K, K), dtype=np.float64)

        for i, h_i in enumerate(self.horizons):
            for j, h_j in enumerate(self.horizons):
                # Overlap fraction: max(0, min(tau_i, tau_j) / max(tau_i, tau_j))
                tau_i = feedback_latencies.get(h_i, h_i)
                tau_j = feedback_latencies.get(h_j, h_j)
                frac = min(tau_i, tau_j) / max(tau_i, tau_j)
                overlap_matrix[i, j] = frac

        eigenvals = np.linalg.eigvalsh(overlap_matrix)
        lambda_max = float(np.max(eigenvals))
        return lambda_max

    def evaluate_bound(
        self,
        horizon: int,
        empirical_crps: float,
        shared_state_history: np.ndarray,
        observations: np.ndarray,
        oracle_crps: float | Dict[int, float] = 0.5,
        loss_scale: Optional[float] = None,
    ) -> PACBayesBoundResult:
        """Evaluates Theorem 2 for a specific horizon h over T steps."""
        T = len(observations)
        if T <= 0:
            raise ValueError("Time horizon T must be positive.")

        if isinstance(oracle_crps, dict):
            h_oracle = float(oracle_crps.get(horizon, 0.5))
        else:
            h_oracle = float(oracle_crps)

        obs = np.asarray(observations, dtype=np.float64)
        if loss_scale is None:
            # Effective loss scale based on observation dispersion (2 * std or IQR)
            loss_scale = max(1.0, float(np.std(obs) * 2.5))

        kl_div = self.compute_manifold_kl_divergence(shared_state_history)
        v_t = self.estimate_total_variation_drift(observations)
        feedback_latencies = {h: h for h in self.horizons}
        lambda_overlap = self.compute_overlap_spectral_norm(horizon, feedback_latencies)

        # Term 1: PAC-Bayes complexity term scaled by loss bound B
        # B * sqrt( (KL + log(2*sqrt(T)/delta)) / (2*T) )
        log_term = math.log((2.0 * math.sqrt(T)) / self.delta)
        complexity_penalty = loss_scale * math.sqrt(max(0.0, kl_div + log_term) / (2.0 * T))

        # Term 2: Stale feedback drift penalty
        # C_h * (tau_h * V_T) / T
        tau_h = float(horizon)
        drift_penalty = self.crps_lipschitz_c * (tau_h * v_t) / float(T)

        # Term 3: Multi-horizon overlap noise penalty
        # lambda_max(C_overlap) * loss_scale / (2 * sqrt(T))
        overlap_penalty = (lambda_overlap * loss_scale) / (2.0 * math.sqrt(T))

        # Theoretical Upper Bound
        theoretical_bound = h_oracle + complexity_penalty + drift_penalty + overlap_penalty
        slack = theoretical_bound - empirical_crps

        return PACBayesBoundResult(
            horizon=horizon,
            time_horizon_T=T,
            empirical_risk=float(empirical_crps),
            theoretical_upper_bound=float(theoretical_bound),
            kl_divergence=float(kl_div),
            complexity_penalty=float(complexity_penalty),
            drift_penalty=float(drift_penalty),
            overlap_penalty=float(overlap_penalty),
            bound_slack=float(slack),
            bound_holds=bool(slack >= -1e-6),
        )

    def evaluate_suite(
        self,
        empirical_crps_by_horizon: Dict[int, float],
        shared_state_history: np.ndarray,
        observations: np.ndarray,
        oracle_crps: float | Dict[int, float] = 0.5,
        loss_scale: Optional[float] = None,
    ) -> Dict[int, PACBayesBoundResult]:
        """Evaluates Theorem 2 across all forecasting horizons."""
        results = {}
        for h in self.horizons:
            crps_val = empirical_crps_by_horizon.get(h, 0.7)
            res = self.evaluate_bound(
                horizon=h,
                empirical_crps=crps_val,
                shared_state_history=shared_state_history,
                observations=observations,
                oracle_crps=oracle_crps,
                loss_scale=loss_scale,
            )
            results[h] = res
        return results
