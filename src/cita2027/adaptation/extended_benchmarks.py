"""
Extended Stress-Testing Benchmarks and Multi-Regime Comparative Suite.

This module provides challenging non-stationary stress-testing regimes:
1. Markov Regime-Switching Drift (transitions between stationary, trend, and shock)
2. Heavy-Tailed Student-t Distribution Shocks (degrees of freedom nu in [2.1, 5.0])
3. GARCH(1,1) Heteroskedastic Volatility Clustering
4. Extended Benchmark Suite comparing all 10 adapters:
   - Baselines: B0 (Zero-Shot), B1 (EWMA Indep), B2 (Static Pooled), B3 (Delay-Agnostic)
   - Competitors: PROCEED, PADRE, StayCALM, AcMCP, RLS-TS
   - Proposed: Unified Overlap-Aware Adapter (H3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cita2027.adaptation.competitors import (
    BaseCompetitor,
    ForecastFeedback,
    get_all_competitors,
)
from cita2027.adaptation.correctors import (
    BaselineB0_Frozen,
    BaselineB1_IndependentEWMA,
    BaselineB2_IndependentLocScale,
    BaselineB3_FullyPooled,
)
from cita2027.adaptation.distribution_metrics import (
    continuous_winkler_score,
    energy_score,
    evaluate_calibration_and_sharpness,
    variogram_score,
)
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector as UnifiedAdapterH3


# ==============================================================================
# 1. Non-Stationary Shift Generators
# ==============================================================================

class MarkovRegimeSwitchingGenerator:
    """
    3-State Hidden Markov Regime-Switching Data Stream.
    
    Regimes:
    - State 0 (Calm): mu = 0.0, sigma = 0.60
    - State 1 (Trending Drift): mu = +1.50 (or -1.50), sigma = 1.00
    - State 2 (Turbulent Crisis): mu = 0.0, sigma = 3.20
    """

    def __init__(
        self,
        transition_matrix: Optional[np.ndarray] = None,
        seed: int = 101,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if transition_matrix is None:
            # High persistence in calm and trend; occasional shock
            self.P = np.array([
                [0.92, 0.06, 0.02],
                [0.08, 0.88, 0.04],
                [0.15, 0.10, 0.75],
            ])
        else:
            self.P = np.asarray(transition_matrix)

        self.regime_params = {
            0: {"mu": 0.0, "sigma": 0.60},
            1: {"mu": 1.5, "sigma": 1.00},
            2: {"mu": 0.0, "sigma": 3.20},
        }

    def generate_series(self, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic prequential series with regime transitions.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (series of shape (length,), hidden_states of shape (length,))
        """
        states = np.zeros(length, dtype=int)
        series = np.zeros(length, dtype=np.float64)

        cur_state = 0
        for t in range(length):
            cur_state = int(self.rng.choice([0, 1, 2], p=self.P[cur_state]))
            states[t] = cur_state
            params = self.regime_params[cur_state]
            series[t] = self.rng.normal(params["mu"], params["sigma"])

        return series, states


class HeavyTailStudentTGenerator:
    """
    Heavy-Tailed Student-t Synthetic Data Generator.
    
    Generates fat-tailed non-Gaussian innovations with degrees of freedom nu.
    For nu <= 3, the process has infinite or near-infinite kurtosis, inducing
    extreme outlier spikes that disrupt unregularized online trackers.
    """

    def __init__(self, degrees_of_freedom: float = 2.8, seed: int = 202):
        self.df = degrees_of_freedom
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_series(self, length: int, drift_rate: float = 0.01) -> np.ndarray:
        # Standard Student-t scaled to unit variance if df > 2
        var_scale = math.sqrt((self.df - 2.0) / self.df) if self.df > 2.0 else 1.0
        raw_innovations = self.rng.standard_t(self.df, size=length) * var_scale

        # Add gradual drift
        drift = drift_rate * np.sin(np.linspace(0, 4 * np.pi, length))
        return raw_innovations + drift


class GarchVolatilityClusterGenerator:
    """
    GARCH(1,1) Volatility Clustering Generator.
    
    Model:
        y_t = sigma_t * epsilon_t,  epsilon_t ~ N(0, 1)
        sigma_t^2 = omega + alpha * y_{t-1}^2 + beta * sigma_{t-1}^2
    """

    def __init__(
        self,
        omega: float = 0.05,
        alpha: float = 0.15,
        beta: float = 0.80,
        seed: int = 303,
    ):
        assert alpha + beta < 1.0, "GARCH process must be covariance-stationary (alpha + beta < 1)"
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng(seed)

    def generate_series(self, length: int) -> np.ndarray:
        y = np.zeros(length)
        sigma2 = np.zeros(length)
        uncond_var = self.omega / (1.0 - self.alpha - self.beta)
        sigma2[0] = uncond_var
        eps = self.rng.normal(size=length)
        y[0] = math.sqrt(sigma2[0]) * eps[0]

        for t in range(1, length):
            sigma2[t] = self.omega + self.alpha * (y[t - 1] ** 2) + self.beta * sigma2[t - 1]
            y[t] = math.sqrt(sigma2[t]) * eps[t]

        return y


# ==============================================================================
# 2. Extended Benchmark Prequential Engine
# ==============================================================================

class ExtendedBenchmarkRunner:
    """
    Comprehensive Prequential Benchmark Engine comparing all adaptation strategies
    under delayed, overlapping feedback.
    """

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        num_samples: int = 100,
        seed: int = 42,
    ):
        self.horizons = sorted(horizons) if horizons else [1, 4, 8, 16]
        self.num_samples = num_samples
        self.seed = seed

    def _init_all_models(self) -> Dict[str, Any]:
        """Instantiate all 10 adapters."""
        models: Dict[str, Any] = {
            # Locked & Classic Baselines
            "B0_ZeroShot": BaselineB0_Frozen(self.horizons),
            "B1_EWMA": BaselineB1_IndependentEWMA(self.horizons, alpha=0.10),
            "B2_StaticPooled": BaselineB2_IndependentLocScale(self.horizons, alpha=0.10),
            "B3_DelayAgnostic": BaselineB3_FullyPooled(self.horizons, alpha=0.08),
            # Proposed Unified Adapter
            "H3_Unified": UnifiedAdapterH3(
                horizons=self.horizons,
                latent_dim=2,
                lr_shared=0.08,
                lr_local=0.03,
                gamma_age=0.08,
                overlap_discount_factor=0.70,
            ),
        }
        # Add Competitors
        competitors = get_all_competitors(self.horizons)
        models.update(competitors)
        return models

    def run_benchmark_on_series(
        self,
        series: np.ndarray,
        regime_name: str = "Regime",
    ) -> pd.DataFrame:
        """
        Run causal prequential evaluation over a time-series stream.
        
        Parameters
        ----------
        series : np.ndarray
            1D array of ground truth series.
        regime_name : str
            Label for this stress scenario.
            
        Returns
        -------
        pd.DataFrame
            Metrics comparison table across all 10 models.
        """
        N = len(series)
        max_h = max(self.horizons)
        eval_T = N - max_h
        assert eval_T > 20, f"Series length {N} too short for horizon {max_h}"

        models = self._init_all_models()
        rng = np.random.default_rng(self.seed)

        # Causal Feedback Queues per model:
        # dict: model_name -> list of ForecastFeedback items
        feedback_queues: Dict[str, List[ForecastFeedback]] = {m: [] for m in models}

        # Per-model evaluation logs:
        # dict: model_name -> list of dict with predictions, true values, etc.
        logs: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}

        for t in range(eval_T):
            # 1. Base Foundation Model forecast simulation at time t
            # Base model provides frozen Monte Carlo forecast centered around recent mean with unit variance
            history_mean = float(np.mean(series[max(0, t - 10) : t + 1])) if t > 0 else float(series[0])
            base_forecasts: Dict[int, np.ndarray] = {}
            for h in self.horizons:
                base_forecasts[h] = rng.normal(loc=history_mean, scale=1.0, size=self.num_samples)

            # 2. Each model makes predictions for all horizons
            for m_name, model in models.items():
                m_preds: Dict[int, np.ndarray] = {}
                for h in self.horizons:
                    if hasattr(model, "predict"):
                        try:
                            adj_samples = model.predict(h, base_forecasts[h], current_step=t)
                        except TypeError:
                            adj_samples = model.predict(h, base_forecasts[h])
                    elif hasattr(model, "adapt"):
                        try:
                            adj_samples = model.adapt(t, h, base_forecasts[h])
                        except TypeError:
                            adj_samples = model.adapt(h, base_forecasts[h])
                    else:
                        adj_samples = base_forecasts[h]
                    m_preds[h] = adj_samples

                # Log prediction for scoring
                y_true_vec = np.array([series[t + h] for h in self.horizons])
                pred_traj = np.stack([m_preds[h] for h in self.horizons], axis=1)  # (S, H)
                logs[m_name].append({
                    "step": t,
                    "y_true": y_true_vec,
                    "pred_trajectories": pred_traj,
                    "base_forecasts": base_forecasts,
                    "preds": m_preds,
                })

                # Schedule causal feedback: arrives at t + h
                for h in self.horizons:
                    fb = ForecastFeedback(
                        origin_step=t,
                        arrival_step=t + h,
                        horizon=h,
                        y_true=float(series[t + h]),
                        y_pred_samples=m_preds[h],
                        base_samples=base_forecasts[h],
                    )
                    feedback_queues[m_name].append(fb)

            # 3. Deliver causal feedback maturing at step t (origin + h == t)
            # Observations maturing at step t were initiated at t - h <= t
            for m_name, model in models.items():
                maturing = [fb for fb in feedback_queues[m_name] if fb.arrival_step == t]
                if maturing:
                    if hasattr(model, "update"):
                        try:
                            model.update(maturing, current_step=t)
                        except TypeError:
                            model.update(maturing)

        # 4. Compute comprehensive proper scores across prequential trajectories
        summary_rows = []
        for m_name in models:
            m_logs = logs[m_name]
            T_eval = len(m_logs)

            es_list = []
            vs05_list = []
            cw_list = []
            crps_per_h = {h: [] for h in self.horizons}
            all_true_seq = []
            all_pred_seq = []

            for entry in m_logs:
                y_true = entry["y_true"]
                trajs = entry["pred_trajectories"]  # (S, H)

                es = energy_score(y_true, trajs)
                vs = variogram_score(y_true, trajs, p=0.5)
                es_list.append(es)
                vs05_list.append(vs)

                for h_idx, h in enumerate(self.horizons):
                    y_val = y_true[h_idx]
                    s_h = trajs[:, h_idx]
                    cw = continuous_winkler_score(y_val, s_h)
                    cw_list.append(cw)

                    crps_val = float(np.mean(np.abs(s_h - y_val))) - 0.5 * float(
                        np.mean(np.abs(s_h[:, None] - s_h[None, :]))
                    )
                    crps_per_h[h].append(crps_val)

                    all_true_seq.append(y_val)
                    all_pred_seq.append(s_h)

            cal_rep = evaluate_calibration_and_sharpness(all_true_seq, all_pred_seq)

            row: Dict[str, Any] = {
                "Regime": regime_name,
                "Model": m_name,
                "Energy_Score": float(np.mean(es_list)),
                "Variogram_05": float(np.mean(vs05_list)),
                "Cont_Winkler": float(np.mean(cw_list)),
                "ECE": cal_rep.expected_calibration_error,
                "Sharpness_90": cal_rep.mean_interval_width_90,
            }
            for h in self.horizons:
                row[f"CRPS_h{h}"] = float(np.mean(crps_per_h[h]))
            row["CRPS_Mean"] = float(np.mean([row[f"CRPS_h{h}"] for h in self.horizons]))

            summary_rows.append(row)

        df_results = pd.DataFrame(summary_rows)
        # Compute relative improvement over B0
        b0_crps = df_results.loc[df_results["Model"] == "B0_ZeroShot", "CRPS_Mean"].values[0]
        df_results["CRPS_Rel_Imp_%"] = (b0_crps - df_results["CRPS_Mean"]) / b0_crps * 100.0

        return df_results


def run_all_extended_scenarios(
    length: int = 140,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Execute extended stress benchmark across Markov, Student-t, and GARCH scenarios.
    
    Returns
    -------
    pd.DataFrame
        Concatenated multi-scenario benchmark report.
    """
    runner = ExtendedBenchmarkRunner(horizons=[1, 4, 8, 16], num_samples=60, seed=seed)
    dfs = []

    # Scenario 1: Markov Regime-Switching
    markov_gen = MarkovRegimeSwitchingGenerator(seed=seed)
    s1, _ = markov_gen.generate_series(length)
    df1 = runner.run_benchmark_on_series(s1, regime_name="Markov_Switching")
    dfs.append(df1)

    # Scenario 2: Student-t Heavy Tails
    t_gen = HeavyTailStudentTGenerator(degrees_of_freedom=2.6, seed=seed)
    s2 = t_gen.generate_series(length)
    df2 = runner.run_benchmark_on_series(s2, regime_name="StudentT_HeavyTail")
    dfs.append(df2)

    # Scenario 3: GARCH(1,1) Volatility Cluster
    garch_gen = GarchVolatilityClusterGenerator(seed=seed)
    s3 = garch_gen.generate_series(length)
    df3 = runner.run_benchmark_on_series(s3, regime_name="GARCH_Volatility")
    dfs.append(df3)

    return pd.concat(dfs, ignore_index=True)
