
"""Validate SPY daily prices and construct causally indexed log returns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


REQUIRED = ("date", "close", "adj_close", "dividends", "stock_splits")


class DataIntegrityError(ValueError):
    """Raised when input data cannot be accepted without undocumented repair."""


@dataclass(frozen=True)
class Boundaries:
    """Exclusive chronological split boundaries."""

    warmup_end_exclusive: str = "2018-01-01"
    development_end_exclusive: str = "2023-01-01"
    validation_end_exclusive: str = "2025-01-01"
    final_test_end_exclusive: str = "2026-11-01"

    def validate(self) -> None:
        values = (
            self.warmup_end_exclusive,
            self.development_end_exclusive,
            self.validation_end_exclusive,
            self.final_test_end_exclusive,
        )
        try:
            bounds = [date.fromisoformat(value) for value in values]
        except ValueError as exc:
            raise DataIntegrityError("Split bounds must be ISO dates") from exc

        if any(a >= b for a, b in zip(bounds, bounds[1:])):
            raise DataIntegrityError(
                "Split boundaries must be strictly increasing"
            )


def validate_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate raw data without sorting, filling, or dropping observations."""
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise DataIntegrityError(f"Missing fields: {missing}")
    if len(frame) < 2:
        raise DataIntegrityError("At least two observations are required")

    result = frame.copy()

    try:
        dates = pd.to_datetime(
            result["date"], format="%Y-%m-%d", errors="raise"
        )
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError("Unparseable date") from exc

    if (
        dates.isna().any()
        or dates.dt.tz is not None
        or not dates.eq(dates.dt.normalize()).all()
        or dates.duplicated().any()
        or not dates.is_monotonic_increasing
    ):
        raise DataIntegrityError(
            "Dates must be strictly ordered and unique, "
            "timezone-naive session labels"
        )

    result["date"] = dates

    for column in ("close", "adj_close", "dividends", "stock_splits"):
        try:
            values = pd.to_numeric(result[column], errors="raise")
            numeric = values.to_numpy(dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise DataIntegrityError(f"Invalid numeric field: {column}") from exc

        if not np.isfinite(numeric).all():
            raise DataIntegrityError(f"Nonfinite values in {column}")

        result[column] = numeric

    if (result[["close", "adj_close"]] <= 0).any().any():
        raise DataIntegrityError("Prices must be strictly positive")

    if (result[["dividends", "stock_splits"]] < 0).any().any():
        raise DataIntegrityError("Corporate actions cannot be negative")

    return result


def check_sessions(
    dates: pd.Series,
    calendar_name: str = "XNYS",
) -> dict:
    """Compare observed dates against the exchange calendar over the full span."""
    import exchange_calendars as xcals

    if dates.empty:
        raise DataIntegrityError("Cannot check an empty date series")

    first = pd.Timestamp(dates.iloc[0]).date()
    last = pd.Timestamp(dates.iloc[-1]).date()

    try:
        # Explicit bounds avoid the library's rolling default calendar range.
        calendar = xcals.get_calendar(
            calendar_name,
            start=str(first),
            end=str(last),
        )
        expected = pd.DatetimeIndex(
            calendar.sessions_in_range(str(first), str(last))
        ).tz_localize(None)
    except (ValueError, OverflowError) as exc:
        raise DataIntegrityError(
            f"Calendar unavailable for requested span: {exc}"
        ) from exc

    observed = pd.DatetimeIndex(dates)
    missing = expected.difference(observed)
    unexpected = observed.difference(expected)

    return {
        "calendar": calendar_name,
        "expected_sessions": len(expected),
        "observed_sessions": len(observed),
        "missing_sessions": [d.date().isoformat() for d in missing],
        "unexpected_sessions": [d.date().isoformat() for d in unexpected],
    }


def build_returns(
    raw: pd.DataFrame,
    boundaries: Boundaries = Boundaries(),
) -> pd.DataFrame:
    """Construct r_t = log(P_t) - log(P_{t-1}) from adjusted daily closes."""
    boundaries.validate()
    data = validate_raw(raw)

    output = data[
        ["date", "close", "adj_close", "dividends", "stock_splits"]
    ].copy()

    output["row_index"] = np.arange(len(output), dtype=np.int64)
    output["log_return"] = np.log(output["adj_close"]).diff()
    output["previous_date"] = output["date"].shift(1)
    output["previous_row_index"] = (
        output["row_index"].shift(1).astype("Int64")
    )

    dates = output["date"]
    conditions = [
        dates < pd.Timestamp(boundaries.warmup_end_exclusive),
        dates < pd.Timestamp(boundaries.development_end_exclusive),
        dates < pd.Timestamp(boundaries.validation_end_exclusive),
        dates < pd.Timestamp(boundaries.final_test_end_exclusive),
    ]

    output["split"] = np.select(
        conditions,
        ["warmup", "development", "validation", "final_test"],
        default="outside_protocol",
    )

    # The first return is undefined; it must never be filled with zero.
    output["is_return_defined"] = output["log_return"].notna()

    return output


def report_integrity(
    raw: pd.DataFrame,
    derived: pd.DataFrame,
    calendar_report: dict,
    *,
    threshold: float = 0.25,
) -> dict:
    """Summarize data integrity without modifying or approving observations."""
    returns = derived["log_return"]
    large = derived.loc[returns.abs() > threshold, "date"]

    actions = raw.loc[
        (raw["dividends"] != 0) | (raw["stock_splits"] != 0),
        "date",
    ]

    adjustment_ratio = np.log(
        raw["adj_close"].to_numpy(dtype=float)
        / raw["close"].to_numpy(dtype=float)
    )
    ratio_changes = np.abs(np.diff(adjustment_ratio)) > 1e-4
    adjustment_dates = raw.loc[np.r_[False, ratio_changes], "date"]

    return {
        "rows_raw": len(raw),
        "rows_with_returns": int(returns.notna().sum()),
        "first_date": derived["date"].iloc[0].date().isoformat(),
        "last_date": derived["date"].iloc[-1].date().isoformat(),
        "split_rows": {
            str(key): int(count)
            for key, count in derived.groupby("split").size().items()
        },
        "corporate_action_dates": [
            value.date().isoformat() for value in actions
        ],
        "adjustment_ratio_change_dates": [
            value.date().isoformat() for value in adjustment_dates
        ],
        "large_log_return_dates": [
            value.date().isoformat() for value in large
        ],
        "return_unit": "decimal log return (not percent)",
        "calendar_check": calendar_report,
        "warnings": [
            "Yahoo Adj Close is retrospectively adjusted; point-in-time validity is not proven.",
            "Missing sessions and extreme returns require review before freeze.",
            "Checkpoint pretraining contamination has not been assessed.",
        ],
    }
