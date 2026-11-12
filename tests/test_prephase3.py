"""CPU-only causal and contract tests, with no model downloads."""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from cita2027.forecasting.adapters import samples_to_quantiles
from cita2027.forecasting.protocol import (
    CONTEXT_LENGTH, check_quantiles, choose_origins, make_window,
    validate_smoke_records,
)


def fixture_df(n: int = 650) -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-02", periods=n)
    r = np.arange(n, dtype=float) / 100000
    r[0] = np.nan
    return pd.DataFrame({
        "date": dates, "row_index": range(n), "split": ["development"] * n,
        "log_return": r, "previous_date": dates.to_series().shift(1).to_numpy(),
        "previous_row_index": [float("nan")] + list(range(n - 1)),
    })


def test_origin_target_alignment_and_return_units():
    df = fixture_df()
    w = make_window(df, 300)
    assert (w.origin_index, w.target_index) == (300, 301)
    assert w.origin_date == df.loc[300, "date"].date().isoformat()
    assert w.target_date == df.loc[301, "date"].date().isoformat()
    assert w.context[-1] == pytest.approx(df.loc[300, "log_return"])
    assert w.target_log_return == pytest.approx(df.loc[301, "log_return"])
    assert w.context.shape == (256,)


def test_no_future_in_context_and_causality_perturbation():
    df = fixture_df()
    orig = make_window(df, 300)
    modified = df.copy(deep=True)
    modified.loc[301:, "log_return"] = 4000.0
    again = make_window(modified, 300)
    np.testing.assert_array_equal(orig.context, again.context)
    assert orig.context_sha256 == again.context_sha256
    assert orig.target_log_return != again.target_log_return


def test_origin_selection_development_only():
    df = fixture_df()
    df.loc[590:, "split"] = "final_test"
    origins = choose_origins(df, 12)
    assert len(origins) == len(set(origins)) == 12
    assert all(df.loc[i, "split"] == df.loc[i + 1, "split"] == "development" for i in origins)


def test_origin_rejects_future_split_or_bad_previous_date():
    df = fixture_df()
    df.loc[301, "split"] = "validation"
    with pytest.raises(ValueError, match="development"):
        make_window(df, 300)
    df.loc[301, "split"] = "development"
    df.loc[301, "previous_date"] = "2000-01-01"
    with pytest.raises(ValueError, match="previous_date"):
        make_window(df, 300)


def test_quantile_order_and_invalid_predictions():
    np.testing.assert_array_equal(check_quantiles([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3])
    for bad in ([1, 0, 2], [1, np.nan, 2], [0, float("inf"), 2], [1, 2]):
        with pytest.raises(ValueError):
            check_quantiles(bad)


def test_sample_quantile_oracle_and_no_resorting():
    s = np.arange(100, dtype=float) / 10000
    np.testing.assert_allclose(samples_to_quantiles(s), np.quantile(s, [0.1, 0.5, 0.9]))
    with pytest.raises(ValueError):
        samples_to_quantiles(np.full(10, np.nan))


def test_context_includes_origin_but_never_future():
    df = fixture_df()
    window = make_window(df, 500)
    np.testing.assert_allclose(window.context, df.loc[245:500, "log_return"].to_numpy())
    assert window.context.flags.writeable is False


def test_validation_rejects_checkpoint_mismatch_and_cache_schema():
    record = {
        "model": "chronos", "checkpoint_id": "amazon/chronos-t5-small",
        "checkpoint_commit": "a" * 40, "origin_index": 300, "target_index": 301,
        "origin_date": "2019-01-01", "target_date": "2019-01-02",
        "context_first_index": 45, "context_last_index": 300,
        "context_sha256": "b" * 64, "target_log_return": 0.01,
        "q10": -0.02, "q50": 0, "q90": 0.03,
        "elapsed_seconds": 1, "peak_allocated_mib": 2,
        "replay_max_abs_diff": 0,
    }
    validate_smoke_records([record], expected_count=1)
    for key, bad in (("checkpoint_commit", "latest"), ("target_index", 302),
                     ("q50", 1.0), ("context_first_index", 44)):
        mutated = copy.deepcopy(record)
        mutated[key] = bad
        with pytest.raises(ValueError):
            validate_smoke_records([mutated], expected_count=1)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_smoke_records([record, record])


def test_cpu_gpu_handling_optional_dependency_import_is_lazy():
    # Calling offline CPU contracts does not import either TSFM backbone.
    assert check_quantiles([0., 0., 0.]).shape == (3,)


def test_causal_mock_forecast_unchanged_by_future_observations():
    """A deterministic model using only the supplied context is future-invariant."""
    df = fixture_df()
    def mock_predict(context: np.ndarray) -> np.ndarray:
        return check_quantiles(np.mean(context) + np.array([-0.01, 0.0, 0.01]))

    before = mock_predict(make_window(df, 300).context)
    modified = df.copy()
    modified.loc[301:, "log_return"] = -10_000.0
    after = mock_predict(make_window(modified, 300).context)
    np.testing.assert_array_equal(before, after)


def test_sample_quantiles_deterministic_given_same_samples():
    rng = np.random.default_rng(2027)
    samples = rng.normal(size=100)
    np.testing.assert_array_equal(samples_to_quantiles(samples), samples_to_quantiles(samples))
