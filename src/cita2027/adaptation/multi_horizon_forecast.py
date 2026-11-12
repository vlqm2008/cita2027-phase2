"""
Multi-Horizon Probabilistic Forecast Generator.

Generates Monte Carlo predictive sample paths across lead times H = {1, ..., H_max}
with strict temporal causality and metadata provenance tracking.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.forecast_cache import ForecastCache, ForecastProvenanceEntry


class MultiHorizonForecaster:
    """
    Standardized Multi-Horizon Probabilistic Forecaster.
    
    Ingests history context x_t = (y_{t-L+1}, ..., y_t) and outputs
    Monte Carlo samples of shape (num_samples,) for each horizon h in horizons.
    """

    def __init__(
        self,
        horizons: List[int],
        num_samples: int = 100,
        model_id: str = "Chronos-T5-small",
        cache: Optional[ForecastCache] = None,
    ):
        self.horizons = sorted(list(horizons))
        self.num_samples = num_samples
        self.model_id = model_id
        self.cache = cache or ForecastCache()

    def forecast_step(
        self,
        history: np.ndarray,
        origin_t: int,
        dataset_name: str = "Dataset",
        series_id: str = "Series0",
        simulated_generator: Optional[Callable[[int, int, np.ndarray], np.ndarray]] = None,
    ) -> Dict[int, ForecastProvenanceEntry]:
        """
        Produce multi-horizon forecasts for all lead times h in H at origin_t.
        
        Returns
        -------
        Dict[int, ForecastProvenanceEntry]
            Mapping from horizon h to structured provenance entry.
        """
        results: Dict[int, ForecastProvenanceEntry] = {}

        for h in self.horizons:
            # Check cache
            cached = self.cache.get_forecast(
                dataset_name=dataset_name,
                series_id=series_id,
                model_id=self.model_id,
                origin_t=origin_t,
                horizon=h,
            )
            if cached is not None:
                results[h] = cached
                continue

            # Generate forecast samples
            target_t = origin_t + h
            availability_t = target_t  # Ground truth becomes available at t + h

            if simulated_generator is not None:
                samples = simulated_generator(origin_t, h, history)
            else:
                # Default causal baseline: auto-regressive Gaussian centered around recent momentum
                loc = float(np.mean(history[-10:])) if len(history) >= 10 else float(history[-1])
                disp = float(np.std(history[-20:])) if len(history) >= 20 else 1.0
                disp = max(0.1, disp)
                # Variance grows with horizon sqrt(h)
                scale = disp * math.sqrt(1.0 + 0.1 * h)
                rng = np.random.default_rng(origin_t * 1000 + h)
                samples = rng.normal(loc=loc, scale=scale, size=self.num_samples)

            loc_stat = float(np.median(samples))
            disp_stat = float(np.std(samples))

            entry = ForecastProvenanceEntry(
                dataset_name=dataset_name,
                series_id=series_id,
                model_id=self.model_id,
                origin_t=origin_t,
                target_t=target_t,
                horizon=h,
                availability_t=availability_t,
                num_samples=len(samples),
                samples=samples,
                base_location=loc_stat,
                base_dispersion=disp_stat,
                metadata={"context_length": len(history)},
            )
            self.cache.store_forecast(entry)
            results[h] = entry

        return results
