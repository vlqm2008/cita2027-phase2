"""Unit and regression tests for Phase 1 (Hypothesis H1) adaptation."""
from __future__ import annotations

import numpy as np
import pytest

from cita2027.adaptation.feedback_queue import CausalFeedbackQueue, ForecastFeedbackItem
from cita2027.adaptation.metrics import crps_empirical_samples, winkler_score, interval_metrics
from cita2027.adaptation.correctors import (
    BaselineB0_Frozen,
    BaselineB1_IndependentEWMA,
    BaselineB2_IndependentLocScale,
    BaselineB3_FullyPooled,
    HypothesisH1_SharedStateCorrector,
)
from cita2027.adaptation.synthetic_shifts import SyntheticShiftScenario
from cita2027.adaptation.experiment_h1 import run_prequential_experiment


def test_causal_feedback_queue_no_leakage():
    queue = CausalFeedbackQueue()
    item1 = ForecastFeedbackItem(origin_t=10, horizon=4, target_t=14, y_true=1.5, base_samples=np.zeros(10))
    item2 = ForecastFeedbackItem(origin_t=10, horizon=8, target_t=18, y_true=2.0, base_samples=np.zeros(10))

    queue.enqueue(item1)
    queue.enqueue(item2)

    # At t = 12: neither should be released
    ready = queue.step_and_release(current_t=12)
    assert len(ready) == 0
    assert queue.pending_count == 2

    # At t = 14: only item1 should be released
    ready = queue.step_and_release(current_t=14)
    assert len(ready) == 1
    assert ready[0].horizon == 4
    assert queue.pending_count == 1

    # At t = 17: item2 still not released
    ready = queue.step_and_release(current_t=17)
    assert len(ready) == 0

    # At t = 18: item2 released
    ready = queue.step_and_release(current_t=18)
    assert len(ready) == 1
    assert ready[0].horizon == 8
    assert queue.pending_count == 0


def test_crps_properties():
    # If samples are concentrated exactly at y, CRPS should be 0
    samples = np.array([2.5] * 20)
    assert crps_empirical_samples(samples, 2.5) == 0.0

    # Dispersed samples
    rng = np.random.default_rng(42)
    norm_samples = rng.normal(0, 1, size=100)
    crps_val = crps_empirical_samples(norm_samples, 0.0)
    assert crps_val > 0.0
    # Expected CRPS for standard normal at 0 is 1/sqrt(pi) - 1/(2*sqrt(pi)) = 1/(2*sqrt(pi)) ~ 0.282
    assert 0.20 < crps_val < 0.35


def test_winkler_score():
    # Inside interval [1.0, 3.0]
    ws_inside = winkler_score(1.0, 3.0, 2.0, alpha=0.20)
    assert ws_inside == 2.0  # width only

    # Below lower (y = 0.5)
    ws_below = winkler_score(1.0, 3.0, 0.5, alpha=0.20)
    assert ws_below == 2.0 + (2.0 / 0.20) * (1.0 - 0.5)  # 2 + 10 * 0.5 = 7.0

    # Above upper (y = 4.0)
    ws_above = winkler_score(1.0, 3.0, 4.0, alpha=0.20)
    assert ws_above == 2.0 + (2.0 / 0.20) * (4.0 - 3.0)  # 2 + 10 * 1.0 = 12.0


def test_shared_state_corrector_mechanics():
    horizons = [1, 4, 8, 16]
    corrector = HypothesisH1_SharedStateCorrector(horizons, latent_dim=2, lr_shared=0.1, lr_local=0.05)

    base = np.array([0.0] * 50)
    # Initially bias is 0
    adapted = corrector.adapt(origin_t=0, horizon=8, base_samples=base)
    np.testing.assert_allclose(adapted, base)

    # Feedback arrives for horizon 1 with positive error
    item = ForecastFeedbackItem(
        origin_t=0, horizon=1, target_t=1, y_true=2.0, base_samples=base
    )
    corrector.update([item])

    # Shared state should have shifted > 0
    assert np.any(corrector.shared_state > 0)

    # Now horizon 8 prediction should have adapted upwards BEFORE horizon 8 feedback arrived!
    adapted_h8 = corrector.adapt(origin_t=1, horizon=8, base_samples=base)
    assert np.mean(adapted_h8) > 0.0


def test_scale_multiplier_positivity():
    horizons = [1, 4]
    corrector = HypothesisH1_SharedStateCorrector(horizons, latent_dim=2)

    base = np.random.randn(50)
    for _ in range(10):
        item = ForecastFeedbackItem(
            origin_t=0, horizon=1, target_t=1, y_true=10.0, base_samples=base
        )
        corrector.update([item])

    adapted = corrector.adapt(origin_t=1, horizon=4, base_samples=base)
    # Verify finite and non-degenerate
    assert np.isfinite(adapted).all()
    assert np.std(adapted) > 0.0


def test_deterministic_replay():
    scenario = SyntheticShiftScenario(horizons=[1, 4], n_samples=30, seed=123)
    data1 = scenario.generate("mean_shift", length=100)

    scenario2 = SyntheticShiftScenario(horizons=[1, 4], n_samples=30, seed=123)
    data2 = scenario2.generate("mean_shift", length=100)

    c1 = HypothesisH1_SharedStateCorrector([1, 4])
    c2 = HypothesisH1_SharedStateCorrector([1, 4])

    r1 = run_prequential_experiment(c1, data1)
    r2 = run_prequential_experiment(c2, data2)

    assert r1["overall_crps"] == pytest.approx(r2["overall_crps"], abs=1e-10)
