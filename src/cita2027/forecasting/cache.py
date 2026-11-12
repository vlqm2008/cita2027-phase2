"""TASK-004A: immutable, causal, per-origin pilot cache contracts.

This module deliberately does not import GPU libraries, TSFMs or calibrators.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from cita2027.forecasting.protocol import CONTEXT_LENGTH, check_quantiles, make_window
from cita2027.utils.provenance import canonical_json_bytes, sha256_file

SCHEMA_VERSION = "forecast-cache-pilot-v1"
PILOT_COUNT_MIN, PILOT_COUNT_MAX = 20, 50
RECORD_FIELDS = (
    "schema_version", "plan_sha256", "model", "checkpoint_id", "checkpoint_commit",
    "checkpoint_weight_sha256", "processed_snapshot_id", "processed_sha256",
    "origin_index", "target_index", "origin_date", "target_date", "context_first_index",
    "context_last_index", "context_sha256", "target_log_return", "q10", "q50", "q90",
    "seed", "context_length", "horizon", "samples", "quantile_levels",
    "target_units", "elapsed_seconds", "peak_allocated_mib", "replay_checked",
    "replay_max_abs_diff", "contamination_free_verified",
    "point_in_time_adjusted_prices_verified",
)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def select_pilot_origins(df: pd.DataFrame, count: int = 24) -> list[int]:
    """Select evenly spaced development-only origin/target pairs using split labels only."""
    if not PILOT_COUNT_MIN <= count <= PILOT_COUNT_MAX:
        raise ValueError("Pilot requires 20-50 origins")
    valid = [i for i in range(CONTEXT_LENGTH, len(df) - 1)
             if df.iloc[i]["split"] == "development"
             and df.iloc[i + 1]["split"] == "development"]
    if len(valid) < count:
        raise ValueError("Not enough development origins")
    chosen = [valid[j] for j in np.linspace(0, len(valid) - 1, count, dtype=int)]
    if len(set(chosen)) != count:
        raise ValueError("Duplicate selected origin")
    return chosen


def make_plan(df: pd.DataFrame, count: int, snapshot_id: str, processed_sha256: str,
              smoke_config_sha256: str, checkpoint_audit_sha256: str, seed: int,
              replay_origins: int, replay_tolerance: float) -> dict:
    """Share an immutable, checkpoint-independent origin list across backbones."""
    if replay_origins < 1 or replay_origins > count or replay_tolerance < 0:
        raise ValueError("Invalid replay settings")
    origins = select_pilot_origins(df, count)
    rows = []
    for index in origins:
        w = make_window(df, index)
        rows.append({"origin_index": index, "target_index": w.target_index,
                     "origin_date": w.origin_date, "target_date": w.target_date,
                     "context_sha256": w.context_sha256})
    return {
        "schema_version": SCHEMA_VERSION, "kind": "development_pilot_plan",
        "source_snapshot_id": snapshot_id, "processed_sha256": processed_sha256,
        "smoke_config_sha256": smoke_config_sha256,
        "checkpoint_audit_sha256": checkpoint_audit_sha256,
        "context_length": 256, "horizon": 1, "quantile_levels": [0.1, 0.5, 0.9],
        "samples": 100, "seed": seed, "seed_policy": "global_seed_plus_origin_index",
        "replay_origins": replay_origins, "replay_tolerance": replay_tolerance,
        "target_units": "daily_log_return_decimal", "count": count, "origins": rows,
        "contamination_free_verified": False,
        "point_in_time_adjusted_prices_verified": False,
    }


def write_once_atomic(path: Path, payload: bytes) -> None:
    """Publish a new immutable path atomically; fail rather than overwrite.

    The per-run lock protects concurrent writers. os.link provides atomic no-clobber
    publication, including Windows on a local NTFS volume. Temp stays on same FS.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".stage-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, path)  # FileExistsError if published by another process.
    finally:
        Path(tmp).unlink(missing_ok=True)


def store_or_verify(path: Path, value: Any) -> str:
    """Publish once; existing bytes must match canonical expected bytes exactly."""
    content = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Immutable plan mismatch: {path}")
    else:
        try:
            write_once_atomic(path, content)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"Concurrent conflicting plan: {path}") from None
    return sha256_file(path)


def validate_record(record: dict, plan: dict, plan_sha: str, model: str,
                    checkpoint_id: str, commit: str, weight_sha: str,
                    df: pd.DataFrame) -> None:
    if set(record) != set(RECORD_FIELDS):
        raise ValueError(f"Record schema mismatch: {set(RECORD_FIELDS) ^ set(record)}")
    invariants = {
        "schema_version": SCHEMA_VERSION, "plan_sha256": plan_sha,
        "model": model, "checkpoint_id": checkpoint_id,
        "checkpoint_commit": commit, "checkpoint_weight_sha256": weight_sha,
        "processed_snapshot_id": plan["source_snapshot_id"],
        "processed_sha256": plan["processed_sha256"],
        "context_length": 256, "horizon": 1, "samples": 100,
        "quantile_levels": [0.1, 0.5, 0.9],
        "target_units": "daily_log_return_decimal",
        "contamination_free_verified": False,
        "point_in_time_adjusted_prices_verified": False,
    }
    for k, v in invariants.items():
        if record[k] != v:
            raise ValueError(f"Record invariant mismatch: {k}")
    origin = record["origin_index"]
    if type(origin) is not int or origin not in {o["origin_index"] for o in plan["origins"]}:
        raise ValueError("Unexpected origin")
    expected = next(o for o in plan["origins"] if o["origin_index"] == origin)
    w = make_window(df, origin)
    if (record["target_index"] != w.target_index
            or record["origin_date"] != w.origin_date
            or record["target_date"] != w.target_date
            or record["context_sha256"] != w.context_sha256
            or record["context_first_index"] != origin - 255
            or record["context_last_index"] != origin):
        raise ValueError("Record violates causal window / dates / context hash")
    if any(record[k] != expected[k] for k in expected):
        raise ValueError("Record mismatches immutable origin plan")
    if float(record["target_log_return"]).hex() != w.target_log_return.hex():
        raise ValueError("Realized target mismatch")
    if record["seed"] != plan["seed"] + origin:
        raise ValueError("Wrong per-origin seed")
    check_quantiles([record["q10"], record["q50"], record["q90"]])
    for key in ("elapsed_seconds", "peak_allocated_mib", "replay_max_abs_diff"):
        if not np.isfinite(record[key]) or record[key] < 0:
            raise ValueError(f"Invalid metric: {key}")
    idx = [o["origin_index"] for o in plan["origins"]].index(origin)
    should_replay = idx < plan["replay_origins"]
    if type(record["replay_checked"]) is not bool or record["replay_checked"] != should_replay:
        raise ValueError("Replay status mismatch")
    if record["replay_max_abs_diff"] > plan["replay_tolerance"]:
        raise ValueError("Replay tolerance exceeded")
    if not should_replay and record["replay_max_abs_diff"] != 0:
        raise ValueError("Unchecked replay must use zero placeholder")


def read_valid_record(path: Path, plan: dict, plan_sha: str, model: str,
                      checkpoint_id: str, commit: str, weight_sha: str,
                      df: pd.DataFrame) -> dict:
    """Reject edited/truncated/foreign records on resume instead of recomputing them."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if path.read_bytes() != canonical_json_bytes(record):
        raise ValueError(f"Noncanonical / modified record bytes: {path}")
    validate_record(record, plan, plan_sha, model, checkpoint_id, commit, weight_sha, df)
    return record


def collect_records(records_dir: Path, plan: dict, plan_sha: str, model: str,
                    checkpoint_id: str, commit: str, weight_sha: str,
                    df: pd.DataFrame) -> list[dict]:
    """Require exact planned file set. Reject unexpected/duplicate origin files."""
    expected = {f'{o["origin_index"]:06d}.json' for o in plan["origins"]}
    actual = {p.name for p in records_dir.glob("*.json")} if records_dir.exists() else set()
    if extra := actual - expected:
        raise ValueError(f"Unexpected/duplicate record files: {sorted(extra)}")
    return [read_valid_record(records_dir / f'{o["origin_index"]:06d}.json',
                              plan, plan_sha, model, checkpoint_id, commit, weight_sha, df)
            for o in plan["origins"] if f'{o["origin_index"]:06d}.json' in actual]


def validate_cross_model(a: list[dict], b: list[dict]) -> None:
    """Require identical causal identities and observed targets for both models."""
    if len(a) != len(b) or len(a) != len({r["origin_index"] for r in a}):
        raise ValueError("Unequal or duplicate origin lists")
    fields = ("plan_sha256", "processed_snapshot_id", "processed_sha256", "origin_index",
              "target_index", "origin_date", "target_date", "context_first_index",
              "context_last_index", "context_sha256", "target_log_return", "seed",
              "context_length", "horizon", "samples", "quantile_levels", "target_units")
    for x, y in zip(a, b):
        if x["model"] == y["model"]:
            raise ValueError("Expected distinct backbones")
        if any(x[k] != y[k] for k in fields):
            raise ValueError("Cross-backbone forecast-origin mismatch")
