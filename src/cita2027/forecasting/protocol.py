"""Causal forecast origins, output contracts, and smoke-cache schema."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd


CONTEXT_LENGTH = 256
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
SMOKE_COLUMNS = (
    "model", "checkpoint_id", "checkpoint_commit", "origin_index", "target_index",
    "origin_date", "target_date", "context_first_index", "context_last_index",
    "context_sha256", "target_log_return", "q10", "q50", "q90",
    "elapsed_seconds", "peak_allocated_mib", "replay_max_abs_diff",
)


@dataclass(frozen=True)
class ForecastWindow:
    origin_index: int
    target_index: int
    origin_date: str
    target_date: str
    context: np.ndarray
    target_log_return: float

    @property
    def context_sha256(self) -> str:
        return hashlib.sha256(np.asarray(self.context, dtype="<f8").tobytes()).hexdigest()


def load_processed(path: str) -> pd.DataFrame:
    """Read the original processed artifact without modifying it."""
    df = pd.read_csv(path)
    needed = {"date", "row_index", "split", "log_return", "adj_close", "previous_date", "previous_row_index"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing processed fields: {sorted(needed - set(df.columns))}")
    if len(df) <= CONTEXT_LENGTH + 1:
        raise ValueError("Insufficient observations for context and next-session target")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if not df["date"].is_monotonic_increasing or df["date"].duplicated().any():
        raise ValueError("Dates must be increasing and unique")
    if not np.array_equal(df["row_index"].to_numpy(), np.arange(len(df))):
        raise ValueError("row_index is not consecutive")
    if not df["log_return"].isna().iloc[0] or df["log_return"].iloc[1:].isna().any():
        raise ValueError("Incorrect undefined-return placement")
    return df


def choose_origins(df: pd.DataFrame, count: int = 12, context_length: int = CONTEXT_LENGTH) -> list[int]:
    """Select fixed, equally spaced development targets without consulting outcomes."""
    if not 10 <= count <= 20:
        raise ValueError("Smoke test must use 10-20 origins")
    valid = [i for i in range(context_length, len(df) - 1)
             if df.iloc[i]["split"] == "development" and df.iloc[i + 1]["split"] == "development"]
    if len(valid) < count:
        raise ValueError("Insufficient development forecast origins")
    return [valid[j] for j in np.linspace(0, len(valid) - 1, count, dtype=int)]


def make_window(df: pd.DataFrame, origin_index: int, context_length: int = CONTEXT_LENGTH) -> ForecastWindow:
    """At origin t predict the return in row t+1 using returns in [t-L+1,t]."""
    if not context_length > 0 or not context_length <= origin_index < len(df) - 1:
        raise ValueError("Invalid origin or context length")
    origin = df.iloc[origin_index]
    target = df.iloc[origin_index + 1]
    if origin["split"] != "development" or target["split"] != "development":
        raise ValueError("Smoke tests may only use development origins and targets")
    if pd.Timestamp(target["previous_date"]) != origin["date"]:
        raise ValueError("Target previous_date does not match origin")
    if int(target["previous_row_index"]) != origin_index:
        raise ValueError("Target previous_row_index does not match origin")
    context = df.iloc[origin_index - context_length + 1:origin_index + 1]["log_return"].to_numpy(dtype=np.float64, copy=True)
    if context.shape != (context_length,) or not np.isfinite(context).all():
        raise ValueError("Invalid context shape or nonfinite values")
    observed = float(target["log_return"])
    if not np.isfinite(observed):
        raise ValueError("Nonfinite target return")
    context.setflags(write=False)
    return ForecastWindow(origin_index, origin_index + 1, origin["date"].date().isoformat(),
                          target["date"].date().isoformat(), context, observed)


def check_quantiles(values: object) -> np.ndarray:
    """Validate predictive return quantiles; never silently sort or clamp."""
    q = np.asarray(values, dtype=np.float64)
    if q.shape != (3,) or not np.isfinite(q).all():
        raise ValueError("Expected three finite return quantiles")
    if not np.all(q[:-1] <= q[1:]):
        raise ValueError("Predictive quantiles are not monotonic")
    return q


def validate_smoke_records(records: list[dict], expected_count: int | None = None) -> None:
    """Reject malformed or misaligned smoke forecast records."""
    if expected_count is not None and len(records) != expected_count:
        raise ValueError("Unexpected record count")
    ids = set()
    for record in records:
        missing = set(SMOKE_COLUMNS) - set(record)
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")
        if record["origin_index"] + 1 != record["target_index"]:
            raise ValueError("Forecast target is not the next row")
        if record["context_last_index"] != record["origin_index"] or (
            record["context_last_index"] - record["context_first_index"] + 1 != CONTEXT_LENGTH
        ):
            raise ValueError("Invalid context row indices")
        if not record["checkpoint_commit"] or len(record["checkpoint_commit"]) != 40:
            raise ValueError("Checkpoint revision must be a full 40-character commit SHA")
        if not record["checkpoint_id"] or not record["context_sha256"]:
            raise ValueError("Missing checkpoint or context identity")
        if pd.Timestamp(record["origin_date"]) >= pd.Timestamp(record["target_date"]):
            raise ValueError("Target date is not after origin")
        check_quantiles([record["q10"], record["q50"], record["q90"]])
        if not np.isfinite(float(record["target_log_return"])):
            raise ValueError("Invalid return unit or nonfinite target")
        if not np.isfinite(float(record["elapsed_seconds"])) or float(record["elapsed_seconds"]) < 0:
            raise ValueError("Invalid runtime")
        if not np.isfinite(float(record["replay_max_abs_diff"])):
            raise ValueError("Invalid replay difference")
        key = (record["model"], record["origin_index"])
        if key in ids:
            raise ValueError("Duplicate model/origin")
        ids.add(key)
