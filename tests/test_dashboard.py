"""Unit tests for Interactive Research Dashboard and REST Simulation Engine."""
from __future__ import annotations

import json
import pytest

from cita2027.adaptation.synthetic_shifts import SyntheticShiftScenario
from cita2027.adaptation.correctors import BaselineB0_Frozen, BaselineB1_IndependentEWMA
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.dynamic_conformal_calibrator import AsynchronousCrossHorizonConformalCalibrator
from cita2027.ui.dashboard import HTML_DASHBOARD_TEMPLATE


def test_html_template_contains_core_elements():
    assert "CITA 2027" in HTML_DASHBOARD_TEMPLATE
    assert "Asynchronous Cross-Horizon Adaptation" in HTML_DASHBOARD_TEMPLATE
    assert "runSimulation" in HTML_DASHBOARD_TEMPLATE
    assert "chart.js" in HTML_DASHBOARD_TEMPLATE


def test_dashboard_simulation_core():
    horizons = [1, 4, 8, 16]
    scenario = SyntheticShiftScenario(horizons=horizons, seed=42)
    data = scenario.generate(regime="joint_shift", length=50)

    m_h3 = HypothesisH3_OverlapAwareCorrector(horizons=horizons, latent_dim=2)
    calibrator = AsynchronousCrossHorizonConformalCalibrator(horizons=horizons, alpha=0.10)

    base_s = data["base_forecasts"][0][4]
    adapted = m_h3.adapt(0, 4, base_s)
    inter = calibrator.calibrate_interval(0, 4, adapted)

    assert adapted.shape == base_s.shape
    assert inter.lower < inter.upper
    assert inter.width > 0
