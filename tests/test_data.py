"""Independent regression tests T01-T10 for Phase 2 data integrity."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cita2027.data.core import (Boundaries, DataIntegrityError, build_returns,
                                check_sessions, report_integrity, validate_raw)
from cita2027.data.cli import process
from cita2027.utils.provenance import sha256_file


def toy(prices=(100., 110., 99.), dates=None):
    if dates is None:
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"][:len(prices)]
    return pd.DataFrame({"date": dates, "close": prices, "adj_close": prices,
                         "dividends": [0.] * len(prices), "stock_splits": [0.] * len(prices)})


def test_t01_order_duplicates():
    assert len(validate_raw(toy())) == 3
    for bad in (toy(dates=["2024-01-02", "2024-01-02", "2024-01-04"]),
                toy(dates=["2024-01-03", "2024-01-02", "2024-01-04"])):
        with pytest.raises(DataIntegrityError, match="ordered and unique"):
            validate_raw(bad)


@pytest.mark.parametrize("column,value", [("close", 0), ("adj_close", -1),
                                            ("close", np.nan), ("adj_close", np.inf)])
def test_t02_positive_finite_prices(column, value):
    frame = toy()
    frame.loc[1, column] = value
    with pytest.raises(DataIntegrityError):
        validate_raw(frame)


def test_t03_independent_log_return_oracle():
    result = build_returns(toy())
    assert math.isnan(result["log_return"].iloc[0])
    np.testing.assert_allclose(result["log_return"].iloc[1:].to_numpy(),
                               [math.log(1.1), math.log(0.9)], rtol=0, atol=1e-14)


def test_t04_target_and_previous_row_alignment():
    result = build_returns(toy())
    assert result["row_index"].tolist() == [0, 1, 2]
    assert result["previous_row_index"].iloc[1:].tolist() == [0, 1]
    assert result["previous_date"].iloc[2] == pd.Timestamp("2024-01-03")


def test_t05_next_trading_session_not_calendar_day():
    frame = toy(dates=["2024-01-04", "2024-01-05", "2024-01-08"])
    result = build_returns(frame)
    assert (result.loc[2, "date"] - result.loc[2, "previous_date"]).days == 3
    assert result.loc[2, "previous_row_index"] == 1


def test_t06_causality_future_price_perturbation():
    original = toy()
    changed = original.copy()
    changed.loc[2, ["adj_close", "close"]] = [900., 900.]
    r0 = build_returns(original)["log_return"]
    r1 = build_returns(changed)["log_return"]
    pd.testing.assert_series_equal(r0.iloc[:2], r1.iloc[:2])
    assert not np.isclose(r0.iloc[2], r1.iloc[2])


def test_t07_adjustments_actions_are_preserved():
    frame = toy()
    frame.loc[1, "dividends"] = 1.0
    frame.loc[0, "adj_close"] = 98.0
    result = build_returns(frame)
    assert result.loc[1, "dividends"] == 1.0
    assert not np.isclose(result.loc[1, "log_return"], math.log(110 / 100))
    report = report_integrity(validate_raw(frame), result, {"missing_sessions": [], "unexpected_sessions": []})
    assert report["corporate_action_dates"] == ["2024-01-03"]


def test_t08_boundaries_and_no_overlap():
    dates = ["2017-12-29", "2018-01-02", "2023-01-03", "2025-01-02"]
    result = build_returns(toy(prices=(100., 101., 102., 103.), dates=dates))
    assert result["split"].tolist() == ["warmup", "development", "validation", "final_test"]
    with pytest.raises(DataIntegrityError):
        Boundaries(warmup_end_exclusive="2024-01-01",
                   development_end_exclusive="2023-01-01").validate()


def test_t09_raw_integrity_hash(tmp_path):
    path = tmp_path / "SPY_demo.csv"
    toy().to_csv(path, index=False)
    digest = sha256_file(path)
    assert len(digest) == 64
    path.write_text(path.read_text() + "\n")
    assert sha256_file(path) != digest


def test_t10_missing_columns_and_small_sample():
    with pytest.raises(DataIntegrityError, match="Missing fields"):
        validate_raw(toy().drop(columns=["adj_close"]))
    with pytest.raises(DataIntegrityError, match="At least two"):
        validate_raw(toy().iloc[:1])
    with pytest.raises(DataIntegrityError, match="Unparseable"):
        validate_raw(toy().assign(date=["bad", "2024-01-03", "2024-01-04"]))


def test_calendar_nyse_holiday_and_weekend():
    valid = pd.Series(pd.to_datetime(["2024-01-12", "2024-01-16"]))
    audit = check_sessions(valid)
    assert audit["missing_sessions"] == []  # Jan 15 MLK holiday
    assert audit["unexpected_sessions"] == []
    weekend = pd.Series(pd.to_datetime(["2024-01-12", "2024-01-13", "2024-01-16"]))
    assert check_sessions(weekend)["unexpected_sessions"] == ["2024-01-13"]


def test_missing_session_is_detected():
    days = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-04"]))
    assert check_sessions(days)["missing_sessions"] == ["2024-01-03"]


def test_preprocessing_never_uses_test_statistics():
    dates = ["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]
    frame = toy((100, 110, 105, 107), dates)
    changed = frame.copy()
    changed.loc[3, ["close", "adj_close"]] = 100000
    a, b = build_returns(frame), build_returns(changed)
    pd.testing.assert_frame_equal(a.iloc[:3], b.iloc[:3])


def test_process_refuses_tampered_snapshot(tmp_path):
    root = tmp_path
    raw_path = root / "snapshot.csv"
    toy().to_csv(raw_path, index=False)
    (root / "snapshot.metadata.json").write_text(json.dumps({
        "raw_sha256": "0" * 64, "snapshot_id": "fake"}))
    cfg = root / "data.yaml"
    cfg.write_text("symbol: SPY\n")
    with pytest.raises(DataIntegrityError, match="SHA-256 mismatch"):
        process(raw_path, cfg, root)


def test_report_uses_only_rows_and_observed_metadata():
    frame = validate_raw(toy())
    result = build_returns(frame)
    report = report_integrity(frame, result,
                               {"missing_sessions": [], "unexpected_sessions": []})
    assert report["rows_raw"] == 3
    assert report["rows_with_returns"] == 2
    assert "not proven" in report["warnings"][0]


def test_integration_toy_snapshot_to_processed_report(tmp_path):
    import yaml
    from cita2027.utils.provenance import write_new_bytes

    root = tmp_path
    snapshot = root / "artifacts" / "raw" / "SPY_toy.csv"
    sample = toy()
    write_new_bytes(snapshot, sample.to_csv(index=False).encode())
    sidecar = snapshot.with_suffix(".metadata.json")
    write_new_bytes(sidecar, json.dumps({"raw_sha256": sha256_file(snapshot),
                                        "snapshot_id": "SPY_toy"}).encode())
    cfg_path = root / "configs" / "data.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg = {
        "symbol": "SPY", "start": "2024-01-02", "end_exclusive": "2024-01-05",
        "exchange_calendar": "XNYS", "strict_calendar": True,
        "max_abs_log_return_warning": 0.25,
        "split_boundaries": {
            "warmup_end_exclusive": "2018-01-01",
            "development_end_exclusive": "2023-01-01",
            "validation_end_exclusive": "2025-01-01",
            "final_test_end_exclusive": "2026-11-01"},
    }
    cfg_path.write_text(yaml.safe_dump(cfg))
    outputs = process(snapshot, cfg_path, root)
    data = pd.read_csv(outputs["processed_file"])
    report = json.loads(Path(outputs["report_file"]).read_text())
    manifest = json.loads(Path(outputs["manifest_file"]).read_text())
    assert len(data) == 3
    assert report["status"] == "PASS"
    assert manifest["processed_sha256"] == sha256_file(Path(outputs["processed_file"]))
    assert not manifest["point_in_time_adjusted_prices_verified"]
    with pytest.raises(FileExistsError):
        process(snapshot, cfg_path, root)


def test_missing_boundary_sessions_not_silently_accepted(tmp_path):
    import yaml
    from cita2027.utils.provenance import write_new_bytes

    root = tmp_path
    raw = root / "raw.csv"
    # Only Jan 3 and 4 provided; Jan 2 is missing from requested range.
    toy((110., 99.), ["2024-01-03", "2024-01-04"]).to_csv(raw, index=False)
    write_new_bytes(raw.with_suffix(".metadata.json"), json.dumps({
        "snapshot_id": "toy", "raw_sha256": sha256_file(raw)}).encode())
    cfg_path = root / "config.yaml"
    cfg = {"symbol": "SPY", "start": "2024-01-02", "end_exclusive": "2024-01-05",
           "exchange_calendar": "XNYS", "strict_calendar": True,
           "max_abs_log_return_warning": 0.25,
           "split_boundaries": vars(Boundaries())}
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(DataIntegrityError, match="Calendar mismatch"):
        process(raw, cfg_path, root)
    reports = list((root / "artifacts" / "reports").glob("*.json"))
    assert len(reports) == 1
    assert "2024-01-02" in json.loads(reports[0].read_text())["calendar_check"]["missing_sessions"]

def test_calendar_with_non_session_configuration_boundaries():
    """Calendar bounds may be weekends; requested sessions must remain complete."""
    import exchange_calendars as xcals

    calendar = xcals.get_calendar(
        "XNYS",
        start="2005-01-01",  # Saturday
        end="2026-10-31",    # Saturday
    )

    required = calendar.sessions_in_range(
        calendar.first_session,
        calendar.last_session,
    )

    assert required[0].strftime("%Y-%m-%d") == "2005-01-03"
    assert required[-1].strftime("%Y-%m-%d") == "2026-10-30"
    assert len(required) == 5492