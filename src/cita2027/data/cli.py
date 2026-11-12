
"""Explicit download and validation commands; network never runs on import."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import pandas as pd
import yaml

from cita2027.data.core import (
    Boundaries,
    DataIntegrityError,
    build_returns,
    check_sessions,
    report_integrity,
    validate_raw,
)
from cita2027.data.download import fetch_yahoo, save_snapshot
from cita2027.utils.provenance import sha256_file, write_json_atomic


def load_config(path: Path) -> dict:
    """Load the SPY-specific Phase 2 data configuration."""
    with path.open(encoding="utf-8") as stream:
        conf = yaml.safe_load(stream)

    if conf["symbol"] != "SPY":
        raise ValueError("This Phase 2 pipeline is explicitly restricted to SPY")

    return conf


def process(raw_path: Path, config_path: Path, root: Path) -> dict:
    """Validate an immutable SPY snapshot and construct daily log returns."""
    conf = load_config(config_path)

    # Verify the immutable raw snapshot before parsing its contents.
    sidecar = raw_path.with_suffix(".metadata.json")
    if not sidecar.is_file():
        raise DataIntegrityError("Missing raw snapshot metadata sidecar")

    source_meta = json.loads(sidecar.read_text(encoding="utf-8"))

    if sha256_file(raw_path) != source_meta["raw_sha256"]:
        raise DataIntegrityError("Raw snapshot SHA-256 mismatch")

    raw = pd.read_csv(raw_path, dtype={"date": "string"})
    validated = validate_raw(raw)

    start = pd.Timestamp(conf["start"])
    end_exclusive = pd.Timestamp(conf["end_exclusive"])
    end_inclusive = end_exclusive - pd.Timedelta(days=1)

    if start >= end_exclusive:
        raise DataIntegrityError(
            "Configured start must be earlier than end_exclusive"
        )

    if validated["date"].iloc[0] < start:
        raise DataIntegrityError("Observation predates configured start")

    if validated["date"].iloc[-1] >= end_exclusive:
        raise DataIntegrityError("Observation on/after exclusive end")

    # Construct the full research-period calendar explicitly.
    # Recent exchange_calendars releases may use a rolling default range
    # that no longer covers historical observations from 2005.
    import exchange_calendars as xcals

    exchange = xcals.get_calendar(
        conf["exchange_calendar"],
        start=start.strftime("%Y-%m-%d"),
        end=end_inclusive.strftime("%Y-%m-%d"),
    )

    # Audit endpoint completeness as well as missing internal sessions.
    # Do not silently truncate the expected calendar to observed dates.
    required = exchange.sessions_in_range(
        exchange.first_session,
        exchange.last_session,
    )

    all_observed = pd.DatetimeIndex(validated["date"])
    required_naive = pd.DatetimeIndex(required).tz_localize(None)

    missing_all = required_naive.difference(all_observed)

    # check_sessions independently inspects the observed date range.
    # Its calendar initialization must also cover the historical dates.
    calendar = check_sessions(
        validated["date"],
        conf["exchange_calendar"],
    )

    calendar["missing_sessions"] = sorted(
        set(calendar["missing_sessions"])
        | {timestamp.date().isoformat() for timestamp in missing_all}
    )

    # Construct returns only after raw data and time indexing validation.
    boundaries = Boundaries(**conf["split_boundaries"])
    derived = build_returns(validated, boundaries)

    report = report_integrity(
        validated,
        derived,
        calendar,
        threshold=conf["max_abs_log_return_warning"],
    )

    report["source_snapshot_id"] = source_meta["snapshot_id"]
    report["source_sha256"] = source_meta["raw_sha256"]
    report["config_sha256"] = sha256_file(config_path)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    report["status"] = (
        "PASS"
        if not (
            calendar["missing_sessions"]
            or calendar["unexpected_sessions"]
            or report["large_log_return_dates"]
        )
        else "REVIEW_REQUIRED"
    )

    sid = source_meta["snapshot_id"]
    reports = root / "artifacts" / "reports"

    # Always retain the integrity report, including failed audits.
    write_json_atomic(reports / f"{sid}.json", report)

    if conf["strict_calendar"] and (
        calendar["missing_sessions"]
        or calendar["unexpected_sessions"]
    ):
        raise DataIntegrityError(
            "Calendar mismatch; inspect report before producing processed data"
        )

    if report["status"] != "PASS":
        raise DataIntegrityError(
            "Integrity warnings; inspect report before accepting dataset"
        )

    # Never overwrite an existing processed dataset.
    out = root / "artifacts" / "processed" / f"{sid}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        raise FileExistsError(
            f"Refusing to overwrite processed data: {out}"
        )

    derived.to_csv(out, index=False, float_format="%.17g")

    manifest = {
        "source_id": sid,
        "source_sha256": source_meta["raw_sha256"],
        "processed_sha256": sha256_file(out),
        "config_sha256": sha256_file(config_path),
        "pipeline": "cita2027.data.cli.process",
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "schema_version": 1,
        "point_in_time_adjusted_prices_verified": False,
        "checkpoint_provenance_verified": False,
    }

    write_json_atomic(
        root / "artifacts" / "manifests" / f"{sid}.json",
        manifest,
    )

    return {
        "processed_file": str(out),
        "report_file": str(reports / f"{sid}.json"),
        "manifest_file": str(
            root / "artifacts" / "manifests" / f"{sid}.json"
        ),
    }


def main() -> None:
    """Provide explicit download and process CLI entry points."""
    parser = argparse.ArgumentParser(
        description="CITA2027 SPY data snapshot and validation"
    )

    parser.add_argument(
        "command",
        choices=["download", "process"],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--raw",
        type=Path,
        help="Snapshot CSV for process command",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "download":
        frame, metadata = fetch_yahoo(
            start=config["start"],
            end_exclusive=config["end_exclusive"],
        )
        print(save_snapshot(frame, metadata, args.root))

    else:
        if args.raw is None:
            parser.error("--raw is required for process")

        result = process(
            args.raw,
            args.config,
            args.root,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
