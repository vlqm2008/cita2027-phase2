"""TASK-004B.1: CPU-only full-development origin-plan and pre-run invariants.

Does not perform any forecasting. The existing TASK-004A pipeline remains untouched.
"""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cita2027.forecasting.cache import digest, store_or_verify
from cita2027.forecasting.protocol import CONTEXT_LENGTH, make_window
from cita2027.utils.provenance import canonical_json_bytes, sha256_file

PLAN_VERSION = "full-development-origin-plan-v1"
# Conservative preflight threshold, NOT a measured cache-size requirement.
MIN_FREE_BYTES = 1 * 1024**3


def eligible_full_origins(df: pd.DataFrame) -> list[int]:
    """Select all adjacent development-only origin/target pairs, chronology only."""
    if not df["date"].is_monotonic_increasing or df["date"].duplicated().any():
        raise ValueError("Invalid chronological ordering")
    dev = np.flatnonzero(df["split"].to_numpy() == "development")
    if len(dev) and not np.array_equal(dev, np.arange(dev[0], dev[-1] + 1)):
        raise ValueError("Noncontiguous development split")
    out = [i for i in range(CONTEXT_LENGTH, len(df) - 1)
           if df.iloc[i]["split"] == "development"
           and df.iloc[i + 1]["split"] == "development"]
    if not out:
        raise ValueError("No eligible development forecast pairs")
    if len(out) != len(set(out)):
        raise ValueError("Duplicate development origin")
    return out


def make_full_plan(df: pd.DataFrame, *, snapshot_id: str, processed_sha256: str,
                   smoke_config_sha256: str, checkpoint_audit_sha256: str,
                   seed: int, replay_origins: int, replay_tolerance: float) -> dict:
    origins = eligible_full_origins(df)
    if seed != 2027 or replay_origins != 2 or replay_tolerance != 1e-5:
        raise ValueError("Accepted seed/replay protocol drift")
    rows = []
    for i in origins:
        w = make_window(df, i)
        if w.target_index != i + 1 or i - 255 < 1:
            raise ValueError("Invalid causal window / undefined first return")
        if not (df.iloc[i]["split"] == df.iloc[i + 1]["split"] == "development"):
            raise ValueError("Cross-split forecast target")
        rows.append({"origin_index": i, "target_index": w.target_index,
                     "origin_date": w.origin_date, "target_date": w.target_date,
                     "context_first_index": i - 255, "context_last_index": i,
                     "context_sha256": w.context_sha256,
                     "seed": seed + i})
    return {
        "schema_version": PLAN_VERSION, "kind": "full_development_not_pilot",
        "source_snapshot_id": snapshot_id, "processed_sha256": processed_sha256,
        "smoke_config_sha256": smoke_config_sha256,
        "checkpoint_audit_sha256": checkpoint_audit_sha256,
        "context_length": 256, "horizon": 1, "samples": 100,
        "quantile_levels": [0.1, 0.5, 0.9],
        "empirical_quantile_method": "numpy_linear",
        "seed": 2027, "seed_policy": "global_seed_plus_origin_index",
        "replay_origins": replay_origins, "replay_tolerance": replay_tolerance,
        "target_units": "daily_log_return_decimal",
        "count": len(rows), "first": rows[0], "last": rows[-1],
        "origins": rows, "contamination_free_verified": False,
        "point_in_time_adjusted_prices_verified": False,
        "pilot_reuse_permitted": False,
    }


def validate_full_plan(plan: dict, df: pd.DataFrame) -> None:
    """Reconstruct from dataset, verify every identity, and reject changed plan bytes."""
    rebuilt = make_full_plan(df, snapshot_id=plan["source_snapshot_id"],
        processed_sha256=plan["processed_sha256"],
        smoke_config_sha256=plan["smoke_config_sha256"],
        checkpoint_audit_sha256=plan["checkpoint_audit_sha256"],
        seed=plan["seed"], replay_origins=plan["replay_origins"],
        replay_tolerance=plan["replay_tolerance"])
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(plan):
        raise ValueError("Full development plan drift / corruption")


def publish_plan(root: Path, plan: dict) -> tuple[Path, str]:
    sha = digest(plan)
    path = root / "artifacts" / "forecast_cache" / "full_development" / "plans" / f"{sha}.json"
    if store_or_verify(path, plan) != sha or sha256_file(path) != sha:
        raise ValueError("Published plan SHA mismatch")
    return path, sha


def preflight_disk(root: Path, min_free_bytes: int = MIN_FREE_BYTES) -> dict:
    anchor = root.resolve()
    if not anchor.is_dir():
        raise ValueError("Repository root does not exist")
    usage = shutil.disk_usage(anchor)
    return {"free_bytes": usage.free, "threshold_bytes": min_free_bytes,
            "status": "PASS" if usage.free >= min_free_bytes else "FAIL"}
