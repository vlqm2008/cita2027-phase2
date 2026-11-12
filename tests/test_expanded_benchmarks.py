"""
Unit Tests for Expanded Real-World Benchmark Generators.
"""

import numpy as np
import pytest

from cita2027.adaptation.expanded_real_benchmarks import ExpandedDomainBenchmarkGenerator


def test_generate_solar_grid():
    solar = ExpandedDomainBenchmarkGenerator.generate_solar_irradiance_grid(
        length=250, num_stations=3, seed=123
    )
    assert len(solar) == 3
    for s_name, arr in solar.items():
        assert len(arr) == 250
        assert np.all(arr >= 0.0), "Solar irradiance must be non-negative"
        # Verify cloud shock reduced values between t=150 and t=230
        baseline_mid = np.mean(arr[50:100])
        shock_mid = np.mean(arr[160:210])
        assert shock_mid < baseline_mid, "Cloud shock must depress irradiance"


def test_generate_forex_rates():
    fx = ExpandedDomainBenchmarkGenerator.generate_forex_rates(
        length=300, pairs=["EURUSD", "USDJPY"], seed=456
    )
    assert "EURUSD" in fx and "USDJPY" in fx
    assert len(fx["EURUSD"]) == 300
    assert np.all(fx["EURUSD"] > 0.0), "FX prices must be positive"
    assert np.all(fx["USDJPY"] > 0.0)


def test_generate_electricity_load():
    ecl = ExpandedDomainBenchmarkGenerator.generate_electricity_load_ecl(length=300, seed=789)
    assert "grid_load_mw" in ecl
    load = ecl["grid_load_mw"]
    assert len(load) == 300
    assert np.all(load > 0.0)
    # Check peak surge
    assert np.max(load[200:270]) > np.max(load[:150])
