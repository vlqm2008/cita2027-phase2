"""Synthetic end-to-end checks of the independent read-only verification script."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from cita2027.data.core import build_returns
from cita2027.utils.provenance import sha256_file


def test_phase2_read_only_verifier_works_and_detects_tampering(tmp_path):
    sid = "SPY_fixture"
    raw_dir = tmp_path / "artifacts" / "raw"
    processed_dir = tmp_path / "artifacts" / "processed"
    manifest_dir = tmp_path / "artifacts" / "manifests"
    report_dir = tmp_path / "artifacts" / "reports"
    config_dir = tmp_path / "configs"
    for d in (raw_dir, processed_dir, manifest_dir, report_dir, config_dir):
        d.mkdir(parents=True)
    dates = pd.bdate_range("2018-01-02", periods=270)
    values = 100.0 * np.exp(np.cumsum(np.full(len(dates), 0.001)))
    frame = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "close": values,
        "adj_close": values, "dividends": np.zeros(len(dates)),
        "stock_splits": np.zeros(len(dates)),
    })
    raw = raw_dir / f"{sid}.csv"
    processed = processed_dir / f"{sid}.csv"
    frame.to_csv(raw, index=False)
    build_returns(frame).to_csv(processed, index=False)
    conf = {"split_boundaries": {
        "warmup_end_exclusive": "2018-01-01",
        "development_end_exclusive": "2023-01-01",
        "validation_end_exclusive": "2025-01-01",
        "final_test_end_exclusive": "2026-11-01",
    }}
    config = config_dir / "data.yaml"
    config.write_text(yaml.safe_dump(conf), encoding="utf-8")
    (raw_dir / f"{sid}.metadata.json").write_text(json.dumps({"snapshot_id": sid, "raw_sha256": sha256_file(raw)}))
    (manifest_dir / f"{sid}.json").write_text(json.dumps({
        "source_id": sid, "source_sha256": sha256_file(raw),
        "processed_sha256": sha256_file(processed), "config_sha256": sha256_file(config),
    }))
    (report_dir / f"{sid}.json").write_text(json.dumps({
        "source_snapshot_id": sid, "source_sha256": sha256_file(raw),
        "config_sha256": sha256_file(config), "status": "PASS",
        "rows_raw": len(frame), "rows_with_returns": len(frame) - 1,
        "split_rows": {"development": len(frame)},
        "calendar_check": {"expected_sessions": len(frame), "observed_sessions": len(frame),
                           "missing_sessions": [], "unexpected_sessions": []},
    }))
    verifier = Path(__file__).resolve().parents[1] / "scripts" / "verify_phase2.py"
    run = subprocess.run([sys.executable, str(verifier), "--root", str(tmp_path),
                          "--snapshot-id", sid], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout)["status"] == "PASS"
    processed.write_text(processed.read_text() + "\n")
    tampered = subprocess.run([sys.executable, str(verifier), "--root", str(tmp_path),
                               "--snapshot-id", sid], capture_output=True, text=True)
    assert tampered.returncode != 0
    assert json.loads(tampered.stdout)["status"] == "FAIL"
