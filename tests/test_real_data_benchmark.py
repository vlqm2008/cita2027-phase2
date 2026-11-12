"""Unit tests for Real-World Multi-Horizon Adaptation Benchmark."""
import numpy as np
import pytest
from pathlib import Path

from cita2027.adaptation.real_data_benchmark import (
    find_data_path,
    load_market_data,
    run_real_prequential_experiment,
)
from cita2027.adaptation.correctors import (
    BaselineB0_Frozen,
    BaselineB1_IndependentEWMA,
    HypothesisH1_SharedStateCorrector,
)
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector


def test_find_data_path():
    spy_path = find_data_path("SPY")
    assert spy_path.exists()
    assert "SPY" in spy_path.name

    btc_path = find_data_path("BTCUSDT")
    assert btc_path.exists()
    assert btc_path.suffix == ".csv"


def test_load_market_data():
    df, eval_df, full_returns = load_market_data("SPY", split="validation", context_length=256)
    assert len(eval_df) > 0
    assert len(full_returns) == len(df)
    assert "log_return" in eval_df.columns


def test_real_prequential_experiment_mock():
    horizons = [1, 4, 8, 16]
    n_origins = 50
    num_samples = 30
    max_h = 16

    np.random.seed(42)
    samples = np.random.normal(0, 0.02, size=(n_origins, num_samples, max_h)).astype(np.float32)
    targets = np.random.normal(0, 0.02, size=(n_origins, max_h)).astype(np.float32)
    dates = [f"2023-01-{i+1:02d}" for i in range(n_origins)]

    mock_data = {
        "samples": samples,
        "targets": targets,
        "dates": dates,
        "horizons": horizons,
        "N_origins": n_origins,
    }

    corrector = HypothesisH3_OverlapAwareCorrector(horizons=horizons, latent_dim=2)
    res = run_real_prequential_experiment(corrector, mock_data)

    assert "summary_by_horizon" in res
    assert "overall_crps" in res
    assert res["overall_crps"] > 0
    for h in horizons:
        assert h in res["summary_by_horizon"]
        assert res["summary_by_horizon"][h]["coverage"] >= 0.0
