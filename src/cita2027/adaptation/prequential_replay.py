"""
Causal Prequential Replay Engine.

Executes the strict causal forecast-update loop:
1. Read history x_t in F_t.
2. Generate base and adapted multi-horizon predictive distributions.
3. Cache forecasts before future targets realize.
4. Advance time to t + 1.
5. Release feedback events whose target timestamps have arrived (tau <= t).
6. Update adapter online states using only matured feedback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import CausalFeedbackQueue, ForecastFeedbackItem
from cita2027.adaptation.forecast_cache import ForecastCache, ForecastProvenanceEntry
from cita2027.adaptation.multi_horizon_forecast import MultiHorizonForecaster


@dataclass
class PrequentialStepResult:
    """Artifact emitted at each prequential timestep."""
    step: int
    y_true_horizon: Dict[int, float]
    base_forecasts: Dict[int, np.ndarray]
    adapted_forecasts: Dict[int, np.ndarray]
    matured_feedback_count: int


class PrequentialReplayEngine:
    """
    Executes causal prequential sequence over arbitrary real or synthetic series.
    """

    def __init__(
        self,
        forecaster: MultiHorizonForecaster,
        adapter: Any,
        horizons: List[int],
    ):
        self.forecaster = forecaster
        self.adapter = adapter
        self.horizons = sorted(list(horizons))
        self.max_h = max(self.horizons)
        self.queue = CausalFeedbackQueue()

    def run_prequential_sequence(
        self,
        series: np.ndarray,
        context_window: int = 30,
        dataset_name: str = "PrequentialSeries",
    ) -> List[PrequentialStepResult]:
        """
        Run causal evaluation loop.
        
        Returns
        -------
        List[PrequentialStepResult]
            Timeline of predictions and ground truths.
        """
        N = len(series)
        eval_T = N - self.max_h
        results: List[PrequentialStepResult] = []

        if hasattr(self.adapter, "reset"):
            self.adapter.reset()

        for t in range(eval_T):
            # 1. History available up to t
            history = series[max(0, t - context_window + 1) : t + 1]

            # 2. Base multi-horizon forecasts
            base_entries = self.forecaster.forecast_step(
                history=history,
                origin_t=t,
                dataset_name=dataset_name,
            )

            # 3. Adaptation transformation
            base_dict: Dict[int, np.ndarray] = {}
            adapted_dict: Dict[int, np.ndarray] = {}
            y_true_dict: Dict[int, float] = {}

            for h in self.horizons:
                entry = base_entries[h]
                base_s = entry.samples
                base_dict[h] = base_s
                y_true = float(series[t + h])
                y_true_dict[h] = y_true

                # Apply adapter
                if hasattr(self.adapter, "adapt"):
                    try:
                        adj_s = self.adapter.adapt(t, h, base_s)
                    except TypeError:
                        adj_s = self.adapter.adapt(h, base_s)
                elif hasattr(self.adapter, "predict"):
                    try:
                        adj_s = self.adapter.predict(h, base_s, current_step=t)
                    except TypeError:
                        adj_s = self.adapter.predict(h, base_s)
                else:
                    adj_s = base_s
                adapted_dict[h] = adj_s

                # Enqueue into causal feedback queue
                fb_item = ForecastFeedbackItem(
                    origin_t=t,
                    horizon=h,
                    target_t=t + h,
                    y_true=y_true,
                    base_samples=base_s,
                    adapted_samples=adj_s,
                )
                self.queue.enqueue(fb_item)

            # 4. Advance time and release matured feedback at current step t
            matured = self.queue.step_and_release(t)

            # 5. Update adapter with matured feedback
            if matured and hasattr(self.adapter, "update"):
                try:
                    self.adapter.update(matured, current_step=t)
                except TypeError:
                    self.adapter.update(matured)

            results.append(
                PrequentialStepResult(
                    step=t,
                    y_true_horizon=y_true_dict,
                    base_forecasts=base_dict,
                    adapted_forecasts=adapted_dict,
                    matured_feedback_count=len(matured),
                )
            )

        return results
