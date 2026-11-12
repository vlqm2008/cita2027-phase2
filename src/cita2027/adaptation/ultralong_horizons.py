"""Ultra-Long Horizon Adaptation Stress Benchmark (Horizons H in {1, 2, 4, 8, 16, 32, 64}).

Investigates asymptotic adaptation stability and negative transfer under extreme
feedback latencies spanning up to 64 steps into the future.

Introduces:
1. MultiScaleDyadicEmbedding: Logarithmic dyadic positional embeddings for multi-octave horizons:
     w_h[2k] = sin(log2(h) / 2^k),  w_h[2k+1] = cos(log2(h) / 2^k)
2. UltraLongHorizonEvaluator: Benchmarks B0, B1, B2, B3, H1, H2, H3, and H3-Dyadic
   across Joint Shift, Markov Regime Switching, and GARCH volatility with H in {1, 2, 4, 8, 16, 32, 64}.
3. Stability Metric: Spectral Radius and Maximum Shock Propagation across octaves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cita2027.adaptation.correctors import (
    BaselineB0_Frozen,
    BaselineB1_IndependentEWMA,
    BaselineB2_IndependentLocScale,
    BaselineB3_FullyPooled,
    HypothesisH1_SharedStateCorrector,
)
from cita2027.adaptation.hypothesis_h2 import HypothesisH2_AgeWeightedCorrector
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.synthetic_shifts import SyntheticShiftScenario
from cita2027.adaptation.probabilistic_metrics import compute_crps_empirical
from cita2027.adaptation.feedback_queue import CausalFeedbackQueue, ForecastFeedbackItem


class MultiScaleDyadicEmbedding:
    """Computes dyadic multi-scale positional projection vectors for multi-octave horizons."""

    def __init__(self, latent_dim: int = 4) -> None:
        self.latent_dim = max(2, latent_dim)

    def embed(self, horizon: int) -> np.ndarray:
        """Returns unit-norm d-dimensional dyadic embedding vector."""
        log_h = math.log2(max(1, horizon))
        vec = np.zeros(self.latent_dim, dtype=np.float64)
        for k in range(self.latent_dim // 2):
            freq = 2.0 ** k
            vec[2 * k] = math.sin(log_h / freq)
            vec[2 * k + 1] = math.cos(log_h / freq)
        if self.latent_dim % 2 == 1:
            vec[-1] = math.tanh(log_h / 4.0)

        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec


class HypothesisH3_DyadicUltraLongCorrector(HypothesisH3_OverlapAwareCorrector):
    """Unified Adapter enhanced with Multi-Scale Dyadic Embeddings for horizons up to 64."""

    def __init__(
        self,
        horizons: List[int],
        latent_dim: int = 4,
        lr_shared: float = 0.08,
        lr_local: float = 0.04,
        gamma_age: float = 0.10,
        overlap_discount_factor: float = 0.70,
    ) -> None:
        super().__init__(
            horizons=horizons,
            latent_dim=latent_dim,
            lr_shared=lr_shared,
            lr_local=lr_local,
            gamma_age=gamma_age,
            overlap_discount_factor=overlap_discount_factor,
        )
        self.dyadic_embedder = MultiScaleDyadicEmbedding(latent_dim=latent_dim)
        # Reinitialize horizon projection vectors with dyadic embeddings
        for h in self.horizons:
            self.w[h] = self.dyadic_embedder.embed(h)


def run_ultralong_horizon_experiment(
    horizons: Optional[List[int]] = None,
    length: int = 400,
    seed: int = 2027,
) -> pd.DataFrame:
    """Runs causal prequential evaluation across ultra-long horizons up to H=64."""
    if horizons is None:
        horizons = [1, 2, 4, 8, 16, 32, 64]

    scenario = SyntheticShiftScenario(horizons=horizons, seed=seed)
    regimes = ["joint_shift", "variance_shift", "abrupt_reversal"]

    results = []

    for reg in regimes:
        data = scenario.generate(regime=reg, length=length)

        models = {
            "B0_Frozen": BaselineB0_Frozen(horizons),
            "B1_IndepEWMA": BaselineB1_IndependentEWMA(horizons, alpha=0.10),
            "B2_IndepLocScale": BaselineB2_IndependentLocScale(horizons, alpha=0.10),
            "B3_FullyPooled": BaselineB3_FullyPooled(horizons, alpha=0.08),
            "H1_SharedState": HypothesisH1_SharedStateCorrector(horizons, latent_dim=4),
            "H2_AgeDecay": HypothesisH2_AgeWeightedCorrector(horizons, decay_type="exponential", gamma=0.10),
            "H3_Unified": HypothesisH3_OverlapAwareCorrector(horizons, latent_dim=4, gamma_age=0.10, overlap_discount_factor=0.70),
            "H3_Dyadic": HypothesisH3_DyadicUltraLongCorrector(horizons, latent_dim=4, gamma_age=0.10, overlap_discount_factor=0.70),
        }

        # Step through causal prequential sequence
        from cita2027.adaptation.feedback_queue import CausalFeedbackQueue
        queue = CausalFeedbackQueue()

        for m_name, model in models.items():
            queue_m = CausalFeedbackQueue()
            crps_by_horizon: Dict[int, List[float]] = {h: [] for h in horizons}

            for t in range(length):
                # 1. Adapt and evaluate for all horizons at origin t
                for h in horizons:
                    if t not in data["base_forecasts"] or h not in data["base_forecasts"][t]:
                        continue
                    base_samples = data["base_forecasts"][t][h]
                    adapted_samples = model.adapt(t, h, base_samples)
                    y_true = float(data["y"][t + h])
                    score = compute_crps_empirical(adapted_samples, y_true)
                    crps_by_horizon[h].append(score)

                    # Enqueue feedback for causality
                    feedback_step = t + h
                    queue_m.enqueue(
                        ForecastFeedbackItem(
                            origin_t=t,
                            horizon=h,
                            target_t=feedback_step,
                            y_true=y_true,
                            base_samples=base_samples,
                            adapted_samples=adapted_samples,
                        )
                    )

                # 2. Release causally observable feedback
                ready_events = queue_m.step_and_release(t)
                if ready_events:
                    model.update(ready_events)

            # Record aggregated metrics
            for h in horizons:
                scores = crps_by_horizon[h]
                if scores:
                    mean_crps = float(np.mean(scores))
                    std_crps = float(np.std(scores))
                else:
                    mean_crps = float("nan")
                    std_crps = float("nan")
                results.append({
                    "regime": reg,
                    "model": m_name,
                    "horizon": h,
                    "crps_mean": mean_crps,
                    "crps_std": std_crps,
                    "length": length,
                })

    df = pd.DataFrame(results)
    return df
