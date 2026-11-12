"""Fetch and preserve a new, immutable Yahoo Finance SPY source snapshot."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import uuid

import pandas as pd

from cita2027.utils.provenance import sha256_file, write_json_atomic, write_new_bytes


def normalize_yahoo(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve Close, Adj Close, dividends and splits; never manufacture Adj Close."""
    if isinstance(frame.columns, pd.MultiIndex):
        if len(set(frame.columns.get_level_values(-1))) == 1:
            frame = frame.droplevel(-1, axis=1)
        else:
            raise ValueError("Expected single-ticker Yahoo columns")
    rename = {"Close": "close", "Adj Close": "adj_close",
              "Dividends": "dividends", "Stock Splits": "stock_splits",
              "Open": "open", "High": "high", "Low": "low", "Volume": "volume"}
    frame = frame.rename(columns=rename)
    if "adj_close" not in frame.columns:
        raise ValueError("Adj Close unavailable: do not fall back silently to Close")
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert("America/New_York").tz_localize(None)
    frame.index.name = "date"
    frame = frame.reset_index()
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    for key in ("dividends", "stock_splits"):
        if key not in frame.columns:
            raise ValueError(f"Yahoo did not return {key}; actions=True is required")
    return frame


def fetch_yahoo(*, start: str, end_exclusive: str) -> tuple[pd.DataFrame, dict]:
    """Network operation. Must run only on a machine with network access."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install pip install -e '.[download]' first") from exc
    ticker = yf.Ticker("SPY")
    frame = ticker.history(start=start, end=end_exclusive, interval="1d", auto_adjust=False,
                           actions=True, repair=False, keepna=True, rounding=False,
                           raise_errors=True)
    if frame.empty:
        raise RuntimeError("Yahoo returned no SPY observations")
    data = normalize_yahoo(frame)
    metadata = {"yfinance_version": yf.__version__, "source": "Yahoo Finance via yfinance",
                "ticker": "SPY", "start_inclusive": start,
                "end_exclusive": end_exclusive, "auto_adjust": False, "repair": False,
                "keepna": True, "actions": True,
                "exchange_time_zone": "America/New_York",
                "history_metadata": ticker.get_history_metadata()}
    return data, metadata


def save_snapshot(data: pd.DataFrame, metadata: dict, root: Path) -> Path:
    """Write a newly named raw CSV and sidecar; never overwrite a prior snapshot."""
    now = datetime.now(timezone.utc)
    sid = f"SPY_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"
    path = root / "artifacts" / "raw" / f"{sid}.csv"
    payload = data.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")
    write_new_bytes(path, payload)
    manifest = {**metadata, "snapshot_id": sid, "retrieved_at_utc": now.isoformat(),
                "raw_filename": path.name, "raw_sha256": sha256_file(path),
                "immutable_policy": "Never overwrite raw snapshots. Check SHA-256 on processing."}
    sidecar = path.with_suffix(".metadata.json")
    # This sidecar name is unique because the raw snapshot name is unique.
    write_new_bytes(sidecar, ( __import__("json").dumps(manifest, indent=2, sort_keys=True, default=str) + "\n").encode())
    return path
