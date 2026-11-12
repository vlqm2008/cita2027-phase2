"""TASK-004B.2: separate-interpreter, resumable development-only forecast cache.

Subcommands: preflight, pilot, run, audit. No calibration or test-set access.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from cita2027.forecasting.adapters import seed_inference
from cita2027.forecasting.cache import (
    SCHEMA_VERSION, collect_records, digest, read_valid_record,
    validate_cross_model, validate_record, write_once_atomic, store_or_verify,
)
from cita2027.forecasting.full_plan import make_full_plan, validate_full_plan, preflight_disk
from cita2027.forecasting.protocol import make_window
from cita2027.utils.provenance import canonical_json_bytes, sha256_file
from pilot_forecast_cache import sources, identities, verify_local_checkpoint, version
from plan_full_development import (
    EXPECTED_PROCESSED_SHA, EXPECTED_SMOKE_CONFIG_SHA,
    EXPECTED_CHECKPOINT_AUDIT_SHA, EXPECTED_MODELS,
)

EXPECTED_PLAN_SHA = "39c14b72d7bd82b815757147838e45fe3df56bddf4616367139cfd5661bd089f"
MODELS = ("chronos", "moirai")
COMPLETE_STATUS = "FULL_DEVELOPMENT_COMPLETE_UNAUDITED"
AUDITED_STATUS = "FULL_DEVELOPMENT_FILE_AUDIT_PASS_NOT_SCIENTIFIC_CLEARANCE"


def paths(root: Path, plan_sha: str, model: str) -> Path:
    return root / "artifacts" / "forecast_cache" / "full_development" / "runs" / plan_sha / model


def prepare(root: Path, config: Path, checkpoint_audit: Path):
    """Rebuild all plan entries from the immutable processed dataset on each invocation."""
    conf, df, source, checkpoints = sources(root, config, checkpoint_audit)
    if (source["processed_sha256"] != EXPECTED_PROCESSED_SHA or
            sha256_file(config) != EXPECTED_SMOKE_CONFIG_SHA or
            sha256_file(checkpoint_audit) != EXPECTED_CHECKPOINT_AUDIT_SHA):
        raise ValueError("Previously accepted data/config/checkpoint-audit hashes changed")
    for model in MODELS:
        _, _, rev, files = identities(conf, checkpoints, model)
        if (rev, next(iter(files.values()))) != EXPECTED_MODELS[model]:
            raise ValueError(f"Checkpoint identity drift: {model}")
    full = make_full_plan(df, snapshot_id=conf["source_snapshot_id"],
        processed_sha256=source["processed_sha256"],
        smoke_config_sha256=sha256_file(config),
        checkpoint_audit_sha256=sha256_file(checkpoint_audit),
        seed=int(conf["seed"]), replay_origins=int(conf["replay_origins"]),
        replay_tolerance=float(conf["replay_tolerance"]))
    validate_full_plan(full, df)
    sha = digest(full)
    if sha != EXPECTED_PLAN_SHA or full["count"] != 1258:
        raise ValueError(f"Full plan hash/count drift: {sha} / {full['count']}")
    plan_path = root / "artifacts" / "forecast_cache" / "full_development" / "plans" / f"{sha}.json"
    if not plan_path.is_file() or plan_path.read_bytes() != canonical_json_bytes(full):
        raise ValueError("Published full plan missing or bytes differ")
    if (full["contamination_free_verified"] is not False or
            full["point_in_time_adjusted_prices_verified"] is not False or
            full["pilot_reuse_permitted"] is not False):
        raise ValueError("Research-risk flags drift")
    # Reuse TASK-004A's accepted record schema while preserving the full plan SHA.
    view = {**full, "origins": [
        {k: item[k] for k in ("origin_index", "target_index", "origin_date", "target_date", "context_sha256")}
        for item in full["origins"]]}
    return conf, df, checkpoints, full, view, sha


def identity(conf, checkpoints, model):
    cls, model_id, revision, files = identities(conf, checkpoints, model)
    return cls, model_id, revision, next(iter(files.values())), files


def validate_prefix_records(record_dir, view, sha, model, model_id, rev, weight, df, n):
    """Check all present JSON names and contents, not merely the first n."""
    planned = {f'{o["origin_index"]:06d}.json' for o in view["origins"][:n]}
    actual = {p.name for p in record_dir.glob("*.json")} if record_dir.is_dir() else set()
    if actual - planned:
        raise ValueError(f"Unexpected/duplicate/out-of-scope records: {sorted(actual - planned)[:4]}")
    rows = []
    for o in view["origins"][:n]:
        path = record_dir / f'{o["origin_index"]:06d}.json'
        if path.is_file():
            rows.append(read_valid_record(path, view, sha, model, model_id, rev, weight, df))
    return rows


def make_record(view, sha, model, model_id, revision, weight, window, q, seed, elapsed, peak, checked, diff):
    i = window.origin_index
    return {
        "schema_version": SCHEMA_VERSION, "plan_sha256": sha, "model": model,
        "checkpoint_id": model_id, "checkpoint_commit": revision,
        "checkpoint_weight_sha256": weight,
        "processed_snapshot_id": view["source_snapshot_id"],
        "processed_sha256": view["processed_sha256"],
        "origin_index": i, "target_index": window.target_index,
        "origin_date": window.origin_date, "target_date": window.target_date,
        "context_first_index": i - 255, "context_last_index": i,
        "context_sha256": window.context_sha256,
        "target_log_return": window.target_log_return,
        "q10": float(q[0]), "q50": float(q[1]), "q90": float(q[2]),
        "seed": seed, "context_length": 256, "horizon": 1,
        "samples": 100, "quantile_levels": [0.1, 0.5, 0.9],
        "target_units": "daily_log_return_decimal",
        "elapsed_seconds": elapsed, "peak_allocated_mib": peak,
        "replay_checked": checked, "replay_max_abs_diff": diff,
        "contamination_free_verified": False,
        "point_in_time_adjusted_prices_verified": False,
    }


def manifest_for(records, view, sha, model, model_id, revision, weight, record_dir, csv_sha, runtime):
    return {
        "status": COMPLETE_STATUS, "schema_version": SCHEMA_VERSION,
        "kind": "full_development_not_pilot", "plan_sha256": sha,
        "model": model, "model_id": model_id, "model_revision": revision,
        "weight_sha256": weight, "count": len(records),
        "record_sha256": {f'{r["origin_index"]:06d}.json':
                          sha256_file(record_dir / f'{r["origin_index"]:06d}.json') for r in records},
        "predictions_sha256": csv_sha,
        "processed_snapshot_id": view["source_snapshot_id"],
        "processed_sha256": view["processed_sha256"],
        "smoke_config_sha256": view["smoke_config_sha256"],
        "checkpoint_audit_sha256": view["checkpoint_audit_sha256"],
        "context_length": 256, "horizon": 1, "samples": 100,
        "quantile_levels": [0.1, 0.5, 0.9], "target_units": view["target_units"],
        "seed": 2027, "seed_policy": view["seed_policy"],
        "replay_origins": view["replay_origins"],
        "max_replay_abs_diff": max(r["replay_max_abs_diff"] for r in records),
        "sum_inference_seconds_excluding_loading_replay": sum(r["elapsed_seconds"] for r in records),
        "max_peak_allocated_mib_excluding_model_loading": max(r["peak_allocated_mib"] for r in records),
        **runtime,
        "dependency_compatibility_exception": model == "moirai",
        "contamination_free_verified": False,
        "point_in_time_adjusted_prices_verified": False,
    }


def verify_complete(run_dir, view, sha, model, model_id, revision, weight, df):
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Missing completed full manifest: {model}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if path.read_bytes() != canonical_json_bytes(manifest):
        raise ValueError("Manifest noncanonical/corrupt")
    rows = validate_prefix_records(run_dir / "records", view, sha, model, model_id, revision, weight, df, view["count"])
    if len(rows) != view["count"]:
        raise ValueError("Incomplete records despite completed manifest")
    csv = pd.DataFrame.from_records(rows).to_csv(index=False, float_format="%.17g").encode("utf-8")
    if not (run_dir / "predictions.csv").is_file() or (run_dir / "predictions.csv").read_bytes() != csv:
        raise ValueError("Corrupt or mismatching combined CSV")
    runtime_keys = ("python", "packages", "device", "cuda_compiled", "gpu_name", "gpu_capability")
    runtime = {key: manifest.get(key) for key in runtime_keys}
    if (runtime["device"] != "cuda" or runtime["gpu_capability"] != [12, 0]
            or not runtime["gpu_name"] or not runtime["python"]
            or not isinstance(runtime["packages"], dict)):
        raise ValueError("Invalid saved GPU/runtime provenance")
    expected = manifest_for(rows, view, sha, model, model_id, revision, weight,
                            run_dir / "records", sha256_file(run_dir / "predictions.csv"), runtime)
    # Runtime-created package/environment fields are allowed to differ in current
    # audit interpreter. Verify immutable scientific fields and file hashes.
    if set(manifest) != set(expected):
        raise ValueError("Manifest field set mismatch")
    for key in expected:
        if manifest.get(key) != expected[key]:
            raise ValueError(f"Full manifest corruption / identity mismatch: {key}")
    if manifest["contamination_free_verified"] is not False or manifest["point_in_time_adjusted_prices_verified"] is not False:
        raise ValueError("Scientific-risk flags changed")
    return manifest, rows


def preflight_worker(root, conf, checkpoints, model):
    import torch
    _, model_id, rev, weight, files = identity(conf, checkpoints, model)
    if verify_local_checkpoint(model_id, rev, files) != weight:
        raise ValueError("Local checkpoint mismatch")
    if (not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0)
            or torch.__version__ != "2.9.1+cu128"):
        raise ValueError("CUDA/PyTorch identity drift")
    z = torch.ones((4, 4), device="cuda") @ torch.ones((4, 4), device="cuda")
    torch.cuda.synchronize()
    if not torch.isfinite(z).all().item():
        raise ValueError("CUDA tensor operation failed")
    pkg = "chronos-forecasting" if model == "chronos" else "uni2ts"
    expected = "2.3.2" if model == "chronos" else "1.2.0"
    if importlib.metadata.version(pkg) != expected:
        raise ValueError(f"{model} package version drift")
    if preflight_disk(root)["status"] != "PASS":
        raise ValueError("Insufficient free disk")
    return {"status": "PASS", "model": model, "checkpoint_weight_sha256": weight,
            "torch": torch.__version__, "python": sys.executable,
            "disk_free_bytes": preflight_disk(root)["free_bytes"]}


def worker(root, config, audit, model, count, batch_size, force_limit=None):
    conf, df, checkpoints, full, view, sha = prepare(root, config, audit)
    if not 1 <= count <= full["count"] or batch_size < 1:
        raise ValueError("Invalid count or batch size")
    cls, model_id, rev, weight, files = identity(conf, checkpoints, model)
    run_dir = paths(root, sha, model)
    # Runtime preflight works on resumed nonempty paths, unlike B.1 planning.
    preflight_worker(root, conf, checkpoints, model)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".full.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Run locked; inspect PID before stale-lock removal: {lock_path}") from None
    os.write(fd, f"pid={os.getpid()}\n".encode()); os.close(fd)
    try:
        records_dir = run_dir / "records"
        # Only temporary staging files can be discarded after acquiring lock;
        # published JSON records are never deleted or overwritten.
        for stage in records_dir.glob(".stage-*") if records_dir.exists() else ():
            stage.unlink()
        existing = validate_prefix_records(records_dir, view, sha, model, model_id, rev, weight, df, count)
        if (run_dir / "manifest.json").exists():
            if count != full["count"]:
                raise ValueError("Completed full cache cannot be treated as partial batch")
            verify_complete(run_dir, view, sha, model, model_id, rev, weight, df)
            return {"status": "ALREADY_COMPLETE_VERIFIED", "model": model, "count": len(existing)}
        if (run_dir / "predictions.csv").exists():
            if len(existing) != full["count"]:
                raise ValueError("Orphan CSV with incomplete records")
            expected = pd.DataFrame.from_records(existing).to_csv(index=False, float_format="%.17g").encode("utf-8")
            if (run_dir / "predictions.csv").read_bytes() != expected:
                raise ValueError("Orphan CSV mismatches records")
        adapter = None
        new = 0
        import torch
        try:
            for j, o in enumerate(full["origins"][:count]):
                i = o["origin_index"]
                record_path = records_dir / f"{i:06d}.json"
                if record_path.exists():
                    continue
                if adapter is None:
                    seed_inference(2027)
                    adapter = cls(revision=rev, device="cuda", samples=100)
                w = make_window(df, i)
                current_seed = 2027 + i
                seed_inference(current_seed)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                q = adapter.predict(w.context)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                peak = float(torch.cuda.max_memory_allocated() / 2**20)
                checked = j < view["replay_origins"]
                diff = 0.0
                if checked:
                    seed_inference(current_seed)
                    repeated = adapter.predict(w.context)
                    diff = float(np.max(np.abs(q - repeated)))
                    if not np.allclose(q, repeated, atol=view["replay_tolerance"], rtol=0):
                        raise ValueError(f"Deterministic replay failed: {i}")
                rec = make_record(view, sha, model, model_id, rev, weight, w, q,
                                  current_seed, elapsed, peak, checked, diff)
                validate_record(rec, view, sha, model, model_id, rev, weight, df)
                write_once_atomic(record_path, canonical_json_bytes(rec))
                new += 1
                if new % batch_size == 0 or j == count - 1:
                    print(json.dumps({"progress": model, "new": new, "requested": count,
                                      "latest_origin": i, "batch_size": batch_size}), flush=True)
                if force_limit is not None and new >= force_limit:
                    raise RuntimeError("INTENTIONAL_INTERRUPTION_FOR_TEST")
        finally:
            if adapter is not None:
                del adapter
                torch.cuda.empty_cache()
        rows = validate_prefix_records(records_dir, view, sha, model, model_id, rev, weight, df, count)
        if len(rows) != count:
            raise ValueError("Incomplete requested prefix")
        if count != full["count"]:
            return {"status": "PARTIAL_BATCH_VERIFIED_NOT_COMPLETE", "model": model,
                    "records_in_prefix": len(rows), "new_records": new, "plan_sha256": sha}
        # The entire 1258 records must pass validation before publishing CSV and manifest.
        csv = pd.DataFrame.from_records(rows).to_csv(index=False, float_format="%.17g").encode("utf-8")
        csv_path = run_dir / "predictions.csv"
        if csv_path.exists():
            if csv_path.read_bytes() != csv:
                raise ValueError("Combined CSV differs from records")
        else:
            write_once_atomic(csv_path, csv)
        runtime = {"python": platform.python_version(),
                   "packages": {p: version(p) for p in ("torch", "numpy", "pandas", "chronos-forecasting", "uni2ts", "gluonts", "huggingface-hub", "jax", "toolz")},
                   "device": "cuda", "cuda_compiled": torch.version.cuda,
                   "gpu_name": torch.cuda.get_device_name(0),
                   "gpu_capability": list(torch.cuda.get_device_capability(0))}
        manifest = manifest_for(rows, view, sha, model, model_id, rev, weight,
                                records_dir, sha256_file(csv_path), runtime)
        store_or_verify(run_dir / "manifest.json", manifest)
        verify_complete(run_dir, view, sha, model, model_id, rev, weight, df)
        return {"status": COMPLETE_STATUS, "model": model, "count": len(rows),
                "new_records": new, "manifest": str(run_dir / "manifest.json")}
    finally:
        lock_path.unlink(missing_ok=True)


def audit_cache(root, config, audit, count):
    conf, df, checkpoints, full, view, sha = prepare(root, config, audit)
    if not 1 <= count <= full["count"]:
        raise ValueError("Invalid audit count")
    pairs = []
    details = {}
    for model in MODELS:
        _, model_id, rev, weight, _ = identity(conf, checkpoints, model)
        run_dir = paths(root, sha, model)
        if count == full["count"]:
            manifest, records = verify_complete(run_dir, view, sha, model, model_id, rev, weight, df)
            details[model] = {"records": len(records), "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                              "max_replay_diff": manifest["max_replay_abs_diff"]}
        else:
            if (run_dir / "manifest.json").exists() or (run_dir / "predictions.csv").exists():
                raise ValueError("Partial audit must not masquerade as complete cache")
            records = validate_prefix_records(run_dir / "records", view, sha, model, model_id, rev, weight, df, count)
            if len(records) != count:
                raise ValueError(f"Incomplete partial batch: {model} {len(records)}/{count}")
            details[model] = {"records": len(records), "manifest": None,
                              "max_replay_diff": max(r["replay_max_abs_diff"] for r in records)}
        pairs.append(records)
    validate_cross_model(*pairs)
    report = {"status": AUDITED_STATUS if count == full["count"] else "SMALL_BATCH_AUDIT_PASS_NOT_COMPLETE",
              "plan_sha256": sha, "count_per_model": count,
              "models": details, "cross_model_alignment": "PASS",
              "contamination_free_verified": False,
              "point_in_time_adjusted_prices_verified": False,
              "scientific_clearance": False}
    if count == full["count"]:
        report_path = paths(root, sha, "chronos").parent / "audit_report.json"
        store_or_verify(report_path, report)
        report["audit_report_path"] = str(report_path)
    return report


def gate_path(root: Path, sha: str) -> Path:
    return paths(root, sha, "chronos").parent / "small_batch_gate.json"


def gate_evidence(root, config, audit, count):
    """Bind an accepted two-model batch to immutable per-origin file bytes."""
    conf, df, checkpoints, full, view, sha = prepare(root, config, audit)
    hashes = {}
    paired = []
    for model in MODELS:
        run_dir = paths(root, sha, model)
        _, model_id, rev, weight, _ = identity(conf, checkpoints, model)
        rows = validate_prefix_records(run_dir / "records", view, sha, model, model_id, rev, weight, df, count)
        if len(rows) != count:
            raise ValueError("Incomplete gate records")
        paired.append(rows)
        hashes[model] = {f'{r["origin_index"]:06d}.json': sha256_file(run_dir / "records" / f'{r["origin_index"]:06d}.json') for r in rows}
    validate_cross_model(*paired)
    return {"status": "SMALL_BATCH_GATE_ACCEPTED", "plan_sha256": sha,
            "count_per_model": count, "record_sha256": hashes,
            "contamination_free_verified": False,
            "point_in_time_adjusted_prices_verified": False}


def verify_gate(root, config, audit):
    _, _, _, full, _, sha = prepare(root, config, audit)
    path = gate_path(root, sha)
    if not path.is_file():
        raise ValueError("Small-batch gate missing. Run 'pilot' first; full run forbidden")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if path.read_bytes() != canonical_json_bytes(evidence):
        raise ValueError("Small-batch gate file altered")
    count = evidence.get("count_per_model")
    if type(count) is not int or not 2 <= count <= 50:
        raise ValueError("Invalid batch-gate count")
    if evidence != gate_evidence(root, config, audit, count):
        raise ValueError("Small-batch gate records/hashes no longer match")
    return evidence


def orchestrate(args):
    root = args.root.resolve()
    config = (root / args.config).resolve()
    audit = (root / args.checkpoint_audit).resolve()
    conf, df, checkpoints, full, view, sha = prepare(root, config, audit)
    if args.command == "audit":
        return audit_cache(root, config, audit, args.limit if args.limit else full["count"])
    count = args.limit if args.command == "pilot" else full["count"]
    if args.command == "pilot" and not 2 <= count <= 50:
        raise ValueError("Pilot-prefix batch must contain 2-50 origins")
    if args.command == "run":
        if not args.allow_full:
            raise ValueError("Full 1258-origin execution requires explicit --allow-full")
        verify_gate(root, config, audit)
    if args.command == "preflight":
        count = 0
    results = {}
    for model in MODELS:
        interpreter = root / (".venv" if model == "chronos" else ".venv-moirai") / "Scripts" / "python.exe"
        if not interpreter.is_file():
            raise ValueError(f"Missing interpreter: {interpreter}")
        cmd = [str(interpreter), str(Path(__file__).resolve()), "_worker", "--root", str(root),
               "--config", str(config), "--checkpoint-audit", str(audit), "--model", model]
        if count:
            cmd += ["--count", str(count), "--batch-size", str(args.batch_size)]
        child_env = {**os.environ, "PYTHONPATH": str(root / "src")}
        print(f"[TASK-004B.2] {args.command}: {model}", flush=True)
        completed = subprocess.run(cmd, cwd=root, env=child_env, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"{model} {args.command} FAILED (exit {completed.returncode}); no full audit")
        results[model] = "PASS"
    if args.command == "preflight":
        return {"status": "TWO_ENVIRONMENT_PREFLIGHT_PASS", "models": results,
                "plan_sha256": sha, "scientific_clearance": False}
    report = audit_cache(root, config, audit, count)
    if args.command == "pilot":
        gate = gate_evidence(root, config, audit, count)
        store_or_verify(gate_path(root, sha), gate)
        report["gate_status"] = gate["status"]
        report["gate_path"] = str(gate_path(root, sha))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for cmd in ("preflight", "pilot", "run", "audit", "_worker"):
        sub = subs.add_parser(cmd)
        sub.add_argument("--root", type=Path, default=Path("."))
        sub.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
        sub.add_argument("--checkpoint-audit", type=Path, default=Path("artifacts/checkpoints/hf_audit.json"))
        if cmd in ("pilot", "audit"):
            sub.add_argument("--limit", type=int, default=12 if cmd == "pilot" else None)
        if cmd in ("pilot", "run"):
            sub.add_argument("--batch-size", type=int, default=100)
        if cmd == "run":
            sub.add_argument("--allow-full", action="store_true")
        if cmd == "_worker":
            sub.add_argument("--model", required=True, choices=MODELS)
            sub.add_argument("--count", type=int, default=0)
            sub.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    try:
        if args.command == "_worker":
            root = args.root.resolve()
            cfg = (root / args.config).resolve()
            aud = (root / args.checkpoint_audit).resolve()
            if args.count:
                report = worker(root, cfg, aud, args.model, args.count, args.batch_size)
            else:
                conf, _, checkpoints, _, _, _ = prepare(root, cfg, aud)
                report = preflight_worker(root, conf, checkpoints, args.model)
        else:
            report = orchestrate(args)
        print(json.dumps(report, indent=2), flush=True)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc),
                          "command": args.command}), file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
