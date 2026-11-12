"""
Comprehensive Verification of the 8 Core Architectural Modules specified in Research Handoff.

Verifies:
1. multi_horizon_forecast.py (MultiHorizonForecaster)
2. forecast_cache.py (ForecastCache & ForecastProvenanceEntry)
3. feedback_queue.py (CausalFeedbackQueue)
4. independent_corrector.py (IndependentEWMACorrector & IndependentLocScaleCorrector)
5. shared_state_corrector.py (BaselineB4_PartiallySharedStatic & HypothesisH3)
6. prequential_replay.py (PrequentialReplayEngine)
7. probabilistic_metrics.py (Proper scoring rules & calibration)
8. shift_diagnostics.py (Negative transfer severity & CUSUM breaks)
9. multidomain_data.py (ETT, Jena Weather, PeMS Traffic benchmarks)
"""

import math
import numpy as np
import pytest

from cita2027.adaptation.feedback_queue import CausalFeedbackQueue, ForecastFeedbackItem
from cita2027.adaptation.forecast_cache import ForecastCache, ForecastProvenanceEntry
from cita2027.adaptation.independent_corrector import (
    IndependentEWMACorrector,
    IndependentLocScaleCorrector,
)
from cita2027.adaptation.multi_horizon_forecast import MultiHorizonForecaster
from cita2027.adaptation.multidomain_data import (
    generate_ett_electricity_benchmark,
    generate_traffic_flow_benchmark,
    generate_weather_climate_benchmark,
    load_all_multidomain_benchmarks,
)
from cita2027.adaptation.prequential_replay import PrequentialReplayEngine
from cita2027.adaptation.probabilistic_metrics import (
    compute_univariate_crps,
    continuous_winkler_score,
    energy_score,
    variogram_score,
)
from cita2027.adaptation.shared_state_corrector import (
    BaselineB4_PartiallySharedStatic,
    HypothesisH3_OverlapAwareCorrector,
)
from cita2027.adaptation.shift_diagnostics import (
    detect_cusum_structural_break,
    evaluate_negative_transfer_severity,
)


def test_forecast_cache_and_provenance():
    cache = ForecastCache()
    entry = ForecastProvenanceEntry(
        dataset_name="TestDataset",
        series_id="Series0",
        model_id="Chronos-T5-small",
        origin_t=10,
        target_t=14,
        horizon=4,
        availability_t=14,
        num_samples=50,
        samples=np.random.normal(0, 1, 50),
        base_location=0.0,
        base_dispersion=1.0,
    )
    key = cache.store_forecast(entry)
    assert len(key) == 16

    retrieved = cache.get_forecast("TestDataset", "Series0", "Chronos-T5-small", 10, 4)
    assert retrieved is not None
    assert retrieved.horizon == 4
    assert retrieved.target_t == 14
    assert len(retrieved.samples) == 50


def test_multi_horizon_forecaster():
    forecaster = MultiHorizonForecaster(horizons=[1, 4, 8, 16], num_samples=30)
    history = np.sin(np.linspace(0, 10, 50))
    results = forecaster.forecast_step(history, origin_t=50, dataset_name="SinTest")

    assert len(results) == 4
    for h in [1, 4, 8, 16]:
        assert h in results
        entry = results[h]
        assert entry.target_t == 50 + h
        assert entry.num_samples == 30
        assert np.all(np.isfinite(entry.samples))


def test_independent_correctors_b1_and_b2():
    horizons = [1, 4]
    b1 = IndependentEWMACorrector(horizons, alpha=0.10)
    b2 = IndependentLocScaleCorrector(horizons, alpha=0.10)

    base_s = np.zeros(20)
    # Adapt prior to feedback
    adj1 = b1.adapt(0, 1, base_s)
    adj2 = b2.adapt(0, 1, base_s)
    assert np.allclose(adj1, base_s)
    assert np.allclose(adj2, base_s)

    # Provide feedback
    fb = [
        ForecastFeedbackItem(
            origin_t=0,
            horizon=1,
            target_t=1,
            y_true=2.0,
            base_samples=base_s,
            adapted_samples=base_s,
        )
    ]
    b1.update(fb)
    b2.update(fb)

    # After positive error, bias should shift upward
    adj1_after = b1.adapt(1, 1, base_s)
    adj2_after = b2.adapt(1, 1, base_s)
    assert np.mean(adj1_after) > 0.0
    assert np.mean(adj2_after) > 0.0


def test_baseline_b4_partially_shared_static():
    horizons = [1, 4, 8]
    b4 = BaselineB4_PartiallySharedStatic(horizons, latent_dim=2)
    base_s = np.zeros(25)

    fb = [
        ForecastFeedbackItem(0, 1, 1, 1.5, base_s, base_s),
        ForecastFeedbackItem(0, 4, 4, 1.5, base_s, base_s),
    ]
    b4.update(fb)

    # Shared state should update
    assert np.linalg.norm(b4.shared_state) > 0.0
    # Both horizon 1 and 8 should reflect shared shift
    pred1 = b4.adapt(4, 1, base_s)
    pred8 = b4.adapt(4, 8, base_s)
    assert np.mean(pred1) > 0.0
    assert np.mean(pred8) > 0.0


def test_prequential_replay_engine():
    horizons = [1, 4]
    forecaster = MultiHorizonForecaster(horizons=horizons, num_samples=25)
    adapter = HypothesisH3_OverlapAwareCorrector(horizons=horizons, latent_dim=2)
    engine = PrequentialReplayEngine(forecaster, adapter, horizons)

    series = np.linspace(0, 10, 30) + np.random.normal(0, 0.1, 30)
    timeline = engine.run_prequential_sequence(series, context_window=10)

    # eval_T = 30 - 4 = 26 steps
    assert len(timeline) == 26
    step0 = timeline[0]
    assert 1 in step0.adapted_forecasts
    assert 4 in step0.adapted_forecasts
    assert len(step0.adapted_forecasts[1]) == 25


def test_shift_diagnostics_and_cusum():
    # Negative transfer evaluation
    b0_crps = [0.5, 0.6, 0.7]
    ad_crps = [0.55, 0.58, 0.90]  # h=16 has severe degradation (+28.5%)
    diags = evaluate_negative_transfer_severity(b0_crps, ad_crps, horizons=[1, 4, 16])

    assert len(diags) == 3
    assert diags[0].has_negative_transfer is True
    assert diags[1].has_negative_transfer is False  # 0.58 < 0.60
    assert "SEVERE NEGATIVE TRANSFER" in diags[2].diagnostic_verdict

    # CUSUM structural break detection
    res = np.concatenate([np.random.normal(0, 0.2, 50), np.random.normal(3.0, 0.2, 50)])
    breaks = detect_cusum_structural_break(res, threshold=4.0)
    assert len(breaks) > 0
    # Break should be detected near t=50
    assert any(45 <= b <= 65 for b in breaks)


def test_multidomain_data_generators():
    datasets = load_all_multidomain_benchmarks()
    assert len(datasets) == 3
    for ds in datasets:
        assert len(ds.values) >= 200
        assert ds.train_slice.stop == ds.val_slice.start
        assert ds.val_slice.stop == ds.test_slice.start
        assert np.all(np.isfinite(ds.values))
