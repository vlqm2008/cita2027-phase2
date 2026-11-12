"""
Expanded Real-World Multi-Domain Physical & Financial Benchmark Suites.

Provides synthetic-realistic generators and loaders for:
1. Multi-Station Solar Irradiance Grid (Cloud occlusion structural shocks & diurnal seasonality).
2. International Foreign Exchange Dynamics (Macroeconomic central bank divergence & volatility clusters).
3. Electricity Grid Peak Load Demand (ECL / PJM heatwave load surge).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class ExpandedDomainBenchmarkGenerator:
    """
    Generator for expanded physical and economic benchmarks.
    """

    @staticmethod
    def generate_solar_irradiance_grid(
        length: int = 400,
        num_stations: int = 4,
        seed: int = 101,
    ) -> Dict[str, np.ndarray]:
        """
        Solar irradiance grid with diurnal bell curves and sudden cloud occlusion shocks.
        """
        rng = np.random.default_rng(seed)
        t = np.arange(length)
        day_period = 24.0

        # Base diurnal cycle: max at midday, 0 at night
        diurnal = np.maximum(0.0, np.sin(2.0 * math.pi * t / day_period))

        stations_data = {}
        for s in range(num_stations):
            phase_shift = s * 0.5
            stat_diurnal = np.maximum(
                0.0, np.sin(2.0 * math.pi * (t + phase_shift) / day_period)
            )
            # Baseline irradiance: clear sky GHI ~ 800 W/m^2 normalized to [0, 8.0]
            irradiance = stat_diurnal * 7.5 + rng.normal(0, 0.15, size=length)

            # Cloud occlusion structural break between t=150 and t=230
            # Regional front occludes station s with varying severity
            severity = 0.55 + 0.1 * s
            irradiance[150:230] *= (1.0 - severity)

            stations_data[f"station_{s}"] = np.maximum(0.0, irradiance)

        return stations_data

    @staticmethod
    def generate_forex_rates(
        length: int = 400,
        pairs: List[str] = ["EURUSD", "USDJPY", "GBPUSD"],
        seed: int = 202,
    ) -> Dict[str, np.ndarray]:
        """
        Foreign exchange series with monetary policy structural break and volatility clusters.
        """
        rng = np.random.default_rng(seed)
        fx_data = {}

        for pair in pairs:
            # Baseline random walk returns
            base_vol = 0.005 if "JPY" not in pair else 0.008
            returns = rng.normal(0.0, base_vol, size=length)

            # Central bank rate divergence shift at t=180
            # Induces persistent drift and 3x volatility burst
            if pair == "EURUSD":
                returns[180:280] -= 0.003  # persistent dollar strengthening
                returns[180:280] += rng.normal(0.0, base_vol * 2.5, size=100)
            elif pair == "USDJPY":
                returns[180:280] += 0.005  # sharp JPY depreciation
                returns[180:280] += rng.normal(0.0, base_vol * 3.0, size=100)
            else:
                returns[180:280] -= 0.002
                returns[180:280] += rng.normal(0.0, base_vol * 2.0, size=100)

            # Cumulative price series
            log_price = np.cumsum(returns)
            base_price = 1.08 if pair == "EURUSD" else (150.0 if pair == "USDJPY" else 1.25)
            prices = base_price * np.exp(log_price)
            fx_data[pair] = prices

        return fx_data

    @staticmethod
    def generate_electricity_load_ecl(
        length: int = 400,
        seed: int = 303,
    ) -> Dict[str, np.ndarray]:
        """
        Electricity demand grid (MW) with industrial seasonality and extreme heatwave demand spikes.
        """
        rng = np.random.default_rng(seed)
        t = np.arange(length)
        day_period = 24.0
        week_period = 24.0 * 7.0

        daily_pattern = 10.0 + 4.0 * np.sin(2.0 * math.pi * t / day_period)
        weekly_pattern = 2.0 * np.sin(2.0 * math.pi * t / week_period)
        baseline_load = daily_pattern + weekly_pattern + rng.normal(0, 0.4, size=length)

        # Extreme summer heatwave shock between t=200 and t=270: peak air-conditioning surge (+45% demand)
        baseline_load[200:270] += 5.5 + rng.normal(0, 0.8, size=70)

        return {"grid_load_mw": np.maximum(2.0, baseline_load)}
