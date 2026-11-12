"""TASK-004B.1 CPU-only development-plan tests; no model downloads/inference."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from cita2027.forecasting.full_plan import (
    eligible_full_origins, make_full_plan, preflight_disk, publish_plan,
    validate_full_plan,
)
from cita2027.forecasting.cache import digest, validate_cross_model
from cita2027.forecasting.protocol import make_window
from cita2027.utils.provenance import sha256_file


def fixture_df():
    n = 550
    dates = pd.bdate_range("2017-01-02", periods=n)
    labels = ["warmup"] * 280 + ["development"] * 180 + ["validation"] * 55 + ["final_test"] * 35
    values = np.arange(n, dtype=float) / 100000
    values[0] = np.nan
    return pd.DataFrame({"date": dates, "row_index": range(n), "split": labels,
        "log_return": values, "previous_date": dates.to_series().shift(1).to_numpy(),
        "previous_row_index": [np.nan] + list(range(n - 1))})


def plan(df=None):
    df = fixture_df() if df is None else df
    return make_full_plan(df, snapshot_id="SPY_toy", processed_sha256="a" * 64,
        smoke_config_sha256="b" * 64, checkpoint_audit_sha256="c" * 64,
        seed=2027, replay_origins=2, replay_tolerance=1e-5)


def test_first_last_boundaries_warmup_and_no_validation():
    df = fixture_df()
    p = plan(df)
    assert p["count"] == 179
    assert p["first"]["origin_index"] == 280
    assert p["first"]["context_first_index"] == 25
    assert df.iloc[25]["split"] == "warmup"
    assert p["first"]["target_index"] == 281
    assert p["last"]["origin_index"] == 458
    assert p["last"]["target_index"] == 459
    assert all(df.iloc[o["target_index"]]["split"] == "development" for o in p["origins"])
    assert 459 not in eligible_full_origins(df)


def test_causal_future_perturbation_independent_context():
    df = fixture_df()
    w = make_window(df, 280)
    changed = df.copy()
    changed.loc[281:, "log_return"] += 10000.0
    w2 = make_window(changed, 280)
    assert w.context_sha256 == w2.context_sha256
    np.testing.assert_array_equal(w.context, w2.context)
    assert w.target_log_return != w2.target_log_return


def test_seeds_unique_deterministic_per_origin():
    p = plan()
    assert [o["seed"] for o in p["origins"]] == [2027 + o["origin_index"] for o in p["origins"]]
    assert len({o["seed"] for o in p["origins"]}) == p["count"]
    assert digest(p) == digest(plan())
    with pytest.raises(ValueError, match="protocol drift"):
        make_full_plan(fixture_df(), snapshot_id="toy", processed_sha256="a" * 64,
            smoke_config_sha256="b" * 64, checkpoint_audit_sha256="c" * 64,
            seed=2028, replay_origins=2, replay_tolerance=1e-5)


def test_plan_catches_origin_corruption_and_duplicate():
    p = plan()
    df = fixture_df()
    validate_full_plan(p, df)
    for mutation in (lambda x: x["origins"].__setitem__(1, x["origins"][0]),
                     lambda x: x["last"].__setitem__("target_index", 460),
                     lambda x: x.__setitem__("contamination_free_verified", True)):
        bad = copy.deepcopy(p)
        mutation(bad)
        with pytest.raises(ValueError, match="drift"):
            validate_full_plan(bad, df)


def test_atomic_publish_and_recovery_and_corruption(tmp_path):
    p = plan()
    f, hash0 = publish_plan(tmp_path, p)
    assert hash0 == sha256_file(f)
    assert publish_plan(tmp_path, p) == (f, hash0)
    f.write_text('{"corrupted": true}')
    with pytest.raises(ValueError, match="Immutable plan mismatch"):
        publish_plan(tmp_path, p)


def test_partial_plan_file_is_not_accepted(tmp_path):
    p = plan()
    sha = digest(p)
    f = tmp_path / "artifacts" / "forecast_cache" / "full_development" / "plans" / f"{sha}.json"
    f.parent.mkdir(parents=True)
    f.write_bytes(b'{"incomplete":')
    with pytest.raises(ValueError, match="Immutable plan mismatch"):
        publish_plan(tmp_path, p)


def test_no_pilot_reuse_and_separate_path(tmp_path):
    p = plan()
    f, sha = publish_plan(tmp_path, p)
    assert p["pilot_reuse_permitted"] is False
    assert "full_development" in f.parts and "pilot" not in f.parts
    assert f.name == sha + ".json"


def test_change_split_recomputes_new_plan():
    df = fixture_df()
    first = digest(plan(df))
    changed = df.copy()
    changed.loc[459, "split"] = "validation"
    assert digest(plan(changed)) != first
    assert len(eligible_full_origins(changed)) == 178


def test_missing_history_and_empty_plan_rejected():
    df = fixture_df()
    df.loc[:, "split"] = "warmup"
    with pytest.raises(ValueError, match="No eligible"):
        eligible_full_origins(df)


def test_disk_check(tmp_path):
    assert preflight_disk(tmp_path, min_free_bytes=0)["status"] == "PASS"
    assert preflight_disk(tmp_path, min_free_bytes=10**30)["status"] == "FAIL"


def test_cross_model_alignment_reuses_existing_validator():
    p = plan()
    rows_a, rows_b = [], []
    for r in p["origins"][:3]:
        common = {"plan_sha256": digest(p), "processed_snapshot_id": p["source_snapshot_id"],
            "processed_sha256": p["processed_sha256"], "origin_index": r["origin_index"],
            "target_index": r["target_index"], "origin_date": r["origin_date"],
            "target_date": r["target_date"], "context_first_index": r["context_first_index"],
            "context_last_index": r["context_last_index"], "context_sha256": r["context_sha256"],
            "target_log_return": 0.012, "seed": r["seed"], "context_length": 256,
            "horizon": 1, "samples": 100, "quantile_levels": [0.1, 0.5, 0.9],
            "target_units": "daily_log_return_decimal"}
        rows_a.append({**common, "model": "chronos"})
        rows_b.append({**common, "model": "moirai"})
    validate_cross_model(rows_a, rows_b)
    rows_b[1]["seed"] += 1
    with pytest.raises(ValueError, match="mismatch"):
        validate_cross_model(rows_a, rows_b)


def test_noncontiguous_development_split_rejected():
    df = fixture_df()
    df.loc[360, "split"] = "validation"
    with pytest.raises(ValueError, match="Noncontiguous"):
        eligible_full_origins(df)
