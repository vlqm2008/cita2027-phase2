"""Read-only independent audit of a Phase 2 SPY snapshot and processed artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

from cita2027.utils.provenance import sha256_file


def audit(root: Path, snapshot_id: str) -> dict:
    artifacts = root / "artifacts"
    raw = artifacts / "raw" / f"{snapshot_id}.csv"
    sidecar = artifacts / "raw" / f"{snapshot_id}.metadata.json"
    processed = artifacts / "processed" / f"{snapshot_id}.csv"
    report_path = artifacts / "reports" / f"{snapshot_id}.json"
    manifest_path = artifacts / "manifests" / f"{snapshot_id}.json"
    config_path = root / "configs" / "data.yaml"
    for p in (raw, sidecar, processed, report_path, manifest_path, config_path):
        if not p.is_file():
            raise AssertionError(f"Required file absent: {p}")

    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conf = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert meta["snapshot_id"] == report["source_snapshot_id"] == manifest["source_id"] == snapshot_id
    assert sha256_file(raw) == meta["raw_sha256"] == report["source_sha256"] == manifest["source_sha256"]
    assert sha256_file(processed) == manifest["processed_sha256"]
    assert sha256_file(config_path) == manifest["config_sha256"] == report["config_sha256"]
    assert report["status"] == "PASS"

    r = pd.read_csv(raw, dtype={"date": "string"})
    p = pd.read_csv(processed, dtype={"date": "string", "previous_date": "string"})
    n = len(r)
    assert n >= 257 and n == len(p) == report["rows_raw"]
    assert p["row_index"].tolist() == list(range(n))
    assert p["date"].tolist() == r["date"].tolist()
    dates = pd.to_datetime(p["date"], format="%Y-%m-%d", errors="raise")
    assert dates.is_monotonic_increasing and dates.is_unique
    assert p["previous_date"].iloc[0] is pd.NA or pd.isna(p["previous_date"].iloc[0])
    assert p["previous_date"].iloc[1:].tolist() == p["date"].iloc[:-1].tolist()
    assert pd.isna(p["previous_row_index"].iloc[0])
    np.testing.assert_array_equal(p["previous_row_index"].iloc[1:].to_numpy(dtype=int), np.arange(n - 1))

    prices = pd.to_numeric(r["adj_close"], errors="raise").to_numpy(dtype=np.float64)
    assert (np.isfinite(prices) & (prices > 0)).all()
    theoretical = np.diff(np.log(prices))
    observed = p["log_return"].to_numpy(dtype=np.float64)
    assert np.isnan(observed[0]) and np.isfinite(observed[1:]).all()
    np.testing.assert_allclose(observed[1:], theoretical, rtol=0, atol=1e-12)
    np.testing.assert_allclose(p["adj_close"].to_numpy(dtype=np.float64), prices, rtol=0, atol=1e-12)
    assert int(p["is_return_defined"].sum()) == n - 1 == report["rows_with_returns"]

    bounds = conf["split_boundaries"]
    edges = [bounds[k] for k in (
        "warmup_end_exclusive", "development_end_exclusive",
        "validation_end_exclusive", "final_test_end_exclusive",
    )]
    expected_splits = np.select(
        [dates < pd.Timestamp(x) for x in edges],
        ["warmup", "development", "validation", "final_test"],
        default="outside_protocol",
    )
    np.testing.assert_array_equal(p["split"].to_numpy(), expected_splits)
    counts = p["split"].value_counts().to_dict()
    assert counts == report["split_rows"] and sum(counts.values()) == n
    calendar = report["calendar_check"]
    assert calendar["expected_sessions"] == calendar["observed_sessions"] == n
    assert calendar["missing_sessions"] == calendar["unexpected_sessions"] == []
    return {
        "status": "PASS", "snapshot_id": snapshot_id, "rows": n,
        "defined_returns": n - 1, "raw_sha256": sha256_file(raw),
        "processed_sha256": sha256_file(processed), "config_sha256": sha256_file(config_path),
        "first_date": p["date"].iloc[0], "last_date": p["date"].iloc[-1],
        "split_counts": counts, "max_return_oracle_error": float(np.max(np.abs(observed[1:] - theoretical))),
        "point_in_time_adjusted_prices_verified": False,
        "checkpoint_contamination_excluded": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--snapshot-id", required=True)
    args = ap.parse_args()
    try:
        print(json.dumps(audit(args.root, args.snapshot_id), indent=2))
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
