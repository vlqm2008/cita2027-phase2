"""TASK-004A: generate/audit a development-only frozen TSFM pilot cache.

Run each backbone with its own Python environment. Never calibrates or evaluates.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import yaml

from cita2027.forecasting.adapters import ChronosT5Adapter, Moirai10Adapter, seed_inference
from cita2027.forecasting.cache import (
    SCHEMA_VERSION, collect_records, digest, make_plan, read_valid_record,
    store_or_verify, validate_cross_model, validate_record, write_once_atomic,
)
from cita2027.forecasting.protocol import load_processed, make_window
from cita2027.utils.provenance import canonical_json_bytes, sha256_file


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sources(root: Path, config: Path, audit: Path):
    conf = yaml.safe_load(config.read_text(encoding="utf-8"))
    if (conf["context_length"], conf["horizon"], conf["quantiles"],
            conf["samples"], conf["seed"]) != (256, 1, [0.1, 0.5, 0.9], 100, 2027):
        raise ValueError("Protocol drift from accepted smoke configuration")
    sid = conf["source_snapshot_id"]
    csv = root / "artifacts" / "processed" / f"{sid}.csv"
    manifest = root / "artifacts" / "manifests" / f"{sid}.json"
    source = json.loads(manifest.read_text(encoding="utf-8"))
    if source["source_id"] != sid or sha256_file(csv) != source["processed_sha256"]:
        raise ValueError("Processed SHA / snapshot identity mismatch")
    checkpoints = json.loads(audit.read_text(encoding="utf-8"))
    df = load_processed(str(csv))
    return conf, df, source, checkpoints


def identities(conf: dict, checkpoints: list, model: str):
    cls = {"chronos": ChronosT5Adapter, "moirai": Moirai10Adapter}[model]
    model_id = conf["model_ids"][model]
    if model_id != cls.checkpoint_id:
        raise ValueError("Model ID differs from accepted checkpoint")
    matched = [c for c in checkpoints if c["model_id"] == model_id]
    if len(matched) != 1 or len(matched[0]["commit_sha"]) != 40:
        raise ValueError("Missing immutable checkpoint identity")
    files = {name: meta["lfs_sha256"] for name, meta in matched[0]["files"].items()
             if name.endswith((".safetensors", ".bin")) and meta.get("lfs_sha256")}
    if len(files) != 1:
        raise ValueError("Pilot requires one audited model weight file")
    return cls, model_id, matched[0]["commit_sha"], files


def verify_local_checkpoint(model_id: str, revision: str, files: dict) -> str:
    """Always recheck local checkpoint bytes, including on resumed runs."""
    from huggingface_hub import hf_hub_download
    for name, expected in files.items():
        local = hf_hub_download(repo_id=model_id, filename=name,
                                revision=revision, local_files_only=True)
        if sha256_file(Path(local)) != expected:
            raise ValueError("Checkpoint local SHA-256 mismatches Hub audit")
    return next(iter(files.values()))


def load_plan(root, config, audit, conf, df, source, count):
    plan = make_plan(df, count, conf["source_snapshot_id"],
                     source["processed_sha256"], sha256_file(config),
                     sha256_file(audit), int(conf["seed"]),
                     int(conf["replay_origins"]), float(conf["replay_tolerance"]))
    plan_sha = digest(plan)
    plan_path = root / "artifacts" / "forecast_cache" / "pilot" / "plans" / f"{plan_sha}.json"
    if store_or_verify(plan_path, plan) != plan_sha:
        raise ValueError("Origin-plan serialization/hash mismatch")
    return plan, plan_sha, plan_path


def verify_complete(run_dir: Path, plan, plan_sha, model, model_id, revision,
                    weight_sha, df) -> dict:
    """Read-only verification of completed records, hashes and manifest."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("No completed pilot manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_identity = {
        "status": "PILOT_COMPLETE_UNAUDITED", "schema_version": SCHEMA_VERSION,
        "plan_sha256": plan_sha, "model": model, "model_id": model_id,
        "model_revision": revision, "weight_sha256": weight_sha,
        "processed_snapshot_id": plan["source_snapshot_id"],
        "processed_sha256": plan["processed_sha256"],
        "smoke_config_sha256": plan["smoke_config_sha256"],
        "checkpoint_audit_sha256": plan["checkpoint_audit_sha256"],
        "count": plan["count"], "context_length": 256, "horizon": 1,
        "samples": 100, "quantile_levels": [0.1, 0.5, 0.9],
        "seed": plan["seed"], "seed_policy": plan["seed_policy"],
        "replay_origins": plan["replay_origins"],
        "contamination_free_verified": False,
        "point_in_time_adjusted_prices_verified": False,
        "dependency_compatibility_exception": model == "moirai",
        "device": "cuda", "gpu_capability": [12, 0],
    }
    if any(manifest.get(k) != v for k, v in manifest_identity.items()):
        raise ValueError("Pilot manifest identity/protocol mismatch")
    records = collect_records(run_dir / "records", plan, plan_sha, model,
                              model_id, revision, weight_sha, df)
    if len(records) != plan["count"]:
        raise ValueError("Incomplete pilot records")
    observed = {f'{r["origin_index"]:06d}.json':
                sha256_file(run_dir / "records" / f'{r["origin_index"]:06d}.json')
                for r in records}
    if manifest["record_sha256"] != observed:
        raise ValueError("Per-origin record SHA mismatch")
    if manifest["max_replay_abs_diff"] != max(r["replay_max_abs_diff"] for r in records):
        raise ValueError("Replay summary mismatch")
    if manifest["max_peak_allocated_mib_excluding_model_loading"] != max(r["peak_allocated_mib"] for r in records):
        raise ValueError("Memory summary mismatch")
    if manifest["sum_inference_seconds_excluding_loading_replay"] != sum(r["elapsed_seconds"] for r in records):
        raise ValueError("Runtime summary mismatch")
    if (sha256_file(run_dir / "predictions.csv") != manifest["predictions_sha256"]):
        raise ValueError("Combined CSV SHA mismatch")
    expected = pd.DataFrame.from_records(records).to_csv(index=False, float_format="%.17g").encode("utf-8")
    if (run_dir / "predictions.csv").read_bytes() != expected:
        raise ValueError("CSV differs from immutable JSON records")
    return manifest


def run(args):
    root, config, audit = args.root.resolve(), args.config.resolve(), args.checkpoint_audit.resolve()
    conf, df, source, checkpoints = sources(root, config, audit)
    plan, plan_sha, plan_path = load_plan(root, config, audit, conf, df, source, args.count)
    cls, model_id, revision, weight_files = identities(conf, checkpoints, args.model)
    weight_sha = verify_local_checkpoint(model_id, revision, weight_files)
    run_dir = root / "artifacts" / "forecast_cache" / "pilot" / "runs" / plan_sha / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".pilot.lock"
    # Exclusive lock. On interruption, inspect PID before manually removing stale lock.
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Pilot locked; do not start concurrent job: {lock_path}") from None
    os.write(lock_fd, f"pid={os.getpid()}\n".encode()); os.close(lock_fd)
    try:
        record_dir = run_dir / "records"
        existing = collect_records(record_dir, plan, plan_sha, args.model,
                                   model_id, revision, weight_sha, df)
        if (run_dir / "manifest.json").exists():
            result = verify_complete(run_dir, plan, plan_sha, args.model, model_id,
                                     revision, weight_sha, df)
            print(json.dumps({"status": "ALREADY_COMPLETE_VERIFIED", "records": len(existing),
                              "manifest": str(run_dir / "manifest.json")}, indent=2))
            return
        if (run_dir / "predictions.csv").exists():
            # Crash after atomic CSV publication but before manifest publication:
            # permit recovery only when every record exists and CSV bytes match.
            if len(existing) != plan["count"]:
                raise ValueError("Orphan CSV with incomplete records: inspect manually")
            expected_csv = pd.DataFrame.from_records(existing).to_csv(
                index=False, float_format="%.17g").encode("utf-8")
            if (run_dir / "predictions.csv").read_bytes() != expected_csv:
                raise ValueError("Orphan CSV is inconsistent with immutable records")
        import torch
        if args.device != "cuda" or not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
            raise RuntimeError("TASK-004A pilot requires validated sm_120 CUDA")
        adapter = None
        try:
            for j, entry in enumerate(plan["origins"]):
                i = entry["origin_index"]
                path = record_dir / f"{i:06d}.json"
                if path.is_file():
                    read_valid_record(path, plan, plan_sha, args.model, model_id,
                                      revision, weight_sha, df)
                    print(f"resume {args.model} {j + 1}/{args.count} origin={i}", flush=True)
                    continue
                if adapter is None:
                    seed_inference(int(conf["seed"]))
                    adapter = cls(revision=revision, device=args.device, samples=100)
                window = make_window(df, i)
                current_seed = int(conf["seed"]) + i
                seed_inference(current_seed)
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                t0 = time.perf_counter()
                q = adapter.predict(window.context)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                peak = float(torch.cuda.max_memory_allocated() / 2**20)
                replay_checked = j < int(conf["replay_origins"])
                diff = 0.0
                if replay_checked:
                    seed_inference(current_seed)
                    again = adapter.predict(window.context)
                    diff = float(np.max(np.abs(q - again)))
                    if not np.allclose(q, again, atol=float(conf["replay_tolerance"]), rtol=0):
                        raise RuntimeError(f"Replay mismatch: origin {i}, diff={diff}")
                record = {
                    "schema_version": SCHEMA_VERSION, "plan_sha256": plan_sha,
                    "model": args.model, "checkpoint_id": model_id,
                    "checkpoint_commit": revision, "checkpoint_weight_sha256": weight_sha,
                    "processed_snapshot_id": conf["source_snapshot_id"],
                    "processed_sha256": source["processed_sha256"],
                    "origin_index": i, "target_index": window.target_index,
                    "origin_date": window.origin_date, "target_date": window.target_date,
                    "context_first_index": i - 255, "context_last_index": i,
                    "context_sha256": window.context_sha256,
                    "target_log_return": window.target_log_return,
                    "q10": float(q[0]), "q50": float(q[1]), "q90": float(q[2]),
                    "seed": current_seed, "context_length": 256, "horizon": 1,
                    "samples": 100, "quantile_levels": [0.1, 0.5, 0.9],
                    "target_units": "daily_log_return_decimal",
                    "elapsed_seconds": elapsed, "peak_allocated_mib": peak,
                    "replay_checked": replay_checked, "replay_max_abs_diff": diff,
                    "contamination_free_verified": False,
                    "point_in_time_adjusted_prices_verified": False,
                }
                validate_record(record, plan, plan_sha, args.model, model_id,
                                revision, weight_sha, df)
                write_once_atomic(path, canonical_json_bytes(record))
                print(f"new {args.model} {j + 1}/{args.count} origin={i} "
                      f"elapsed={elapsed:.3f}s peak={peak:.1f}MiB", flush=True)
        finally:
            if adapter is not None:
                del adapter
                torch.cuda.empty_cache()
        records = collect_records(record_dir, plan, plan_sha, args.model,
                                  model_id, revision, weight_sha, df)
        if len(records) != plan["count"]:
            raise RuntimeError("Incomplete pilot, retained records for resume")
        csv_bytes = pd.DataFrame.from_records(records).to_csv(index=False, float_format="%.17g").encode("utf-8")
        if not (run_dir / "predictions.csv").exists():
            write_once_atomic(run_dir / "predictions.csv", csv_bytes)
        elif (run_dir / "predictions.csv").read_bytes() != csv_bytes:
            raise ValueError("Published CSV differs from records")
        manifest = {
            "schema_version": SCHEMA_VERSION, "status": "PILOT_COMPLETE_UNAUDITED",
            "model": args.model, "model_id": model_id, "model_revision": revision,
            "weight_sha256": weight_sha, "plan_sha256": plan_sha,
            "origin_plan_file": str(plan_path.relative_to(root)),
            "processed_snapshot_id": conf["source_snapshot_id"],
            "processed_sha256": source["processed_sha256"],
            "smoke_config_sha256": sha256_file(config),
            "checkpoint_audit_sha256": sha256_file(audit),
            "record_sha256": {f'{r["origin_index"]:06d}.json':
                              sha256_file(record_dir / f'{r["origin_index"]:06d}.json') for r in records},
            "predictions_sha256": sha256_file(run_dir / "predictions.csv"),
            "count": len(records), "context_length": 256, "horizon": 1,
            "samples": 100, "quantile_levels": [0.1, 0.5, 0.9],
            "seed": 2027, "seed_policy": "global_seed_plus_origin_index",
            "replay_origins": conf["replay_origins"],
            "max_replay_abs_diff": max(r["replay_max_abs_diff"] for r in records),
            "sum_inference_seconds_excluding_loading_replay": sum(r["elapsed_seconds"] for r in records),
            "max_peak_allocated_mib_excluding_model_loading": max(r["peak_allocated_mib"] for r in records),
            "python": platform.python_version(),
            "packages": {p: version(p) for p in (
                "torch", "numpy", "pandas", "chronos-forecasting", "uni2ts", "gluonts",
                "huggingface-hub", "jax", "toolz")},
            "device": "cuda", "cuda_compiled": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "contamination_free_verified": False,
            "point_in_time_adjusted_prices_verified": False,
            "dependency_compatibility_exception": args.model == "moirai",
        }
        write_once_atomic(run_dir / "manifest.json", canonical_json_bytes(manifest))
        verify_complete(run_dir, plan, plan_sha, args.model, model_id, revision, weight_sha, df)
        print(json.dumps({"status": "PILOT_COMPLETE_UNAUDITED", "model": args.model,
                          "count": len(records), "manifest": str(run_dir / "manifest.json")}, indent=2))
    finally:
        lock_path.unlink(missing_ok=True)


def audit_pilot(args):
    root, config, audit = args.root.resolve(), args.config.resolve(), args.checkpoint_audit.resolve()
    conf, df, source, checkpoints = sources(root, config, audit)
    # Audit is read-only until all manifests and cross-model identities validate.
    plan = make_plan(df, args.count, conf["source_snapshot_id"], source["processed_sha256"],
                     sha256_file(config), sha256_file(audit), int(conf["seed"]),
                     int(conf["replay_origins"]), float(conf["replay_tolerance"]))
    plan_sha = digest(plan)
    plan_path = root / "artifacts" / "forecast_cache" / "pilot" / "plans" / f"{plan_sha}.json"
    if not plan_path.is_file() or plan_path.read_bytes() != canonical_json_bytes(plan):
        raise ValueError("Immutable origin plan missing or modified")
    results = {}
    pairs = []
    for model in ("chronos", "moirai"):
        _, model_id, revision, files = identities(conf, checkpoints, model)
        weight_sha = next(iter(files.values()))
        run_dir = root / "artifacts" / "forecast_cache" / "pilot" / "runs" / plan_sha / model
        manifest = verify_complete(run_dir, plan, plan_sha, model, model_id, revision, weight_sha, df)
        records = collect_records(run_dir / "records", plan, plan_sha, model, model_id, revision, weight_sha, df)
        if (manifest["contamination_free_verified"] is not False
                or manifest["point_in_time_adjusted_prices_verified"] is not False):
            raise ValueError("Provenance flags must remain unverified")
        results[model] = {"count": len(records), "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                          "max_replay_diff": manifest["max_replay_abs_diff"],
                          "sum_inference_seconds": manifest["sum_inference_seconds_excluding_loading_replay"],
                          "max_peak_allocated_mib": manifest["max_peak_allocated_mib_excluding_model_loading"]}
        pairs.append(records)
    validate_cross_model(*pairs)
    report = {"status": "PILOT_FILE_AUDIT_PASS_NOT_SCIENTIFIC_CLEARANCE",
              "plan_sha256": plan_sha, "models": results,
              "cross_model_alignment": "PASS",
              "contamination_free_verified": False,
              "point_in_time_adjusted_prices_verified": False}
    report_path = root / "artifacts" / "forecast_cache" / "pilot" / "runs" / plan_sha / "audit_report.json"
    store_or_verify(report_path, report)
    print(json.dumps({**report, "audit_report_path": str(report_path)}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("run", "audit"), required=True)
    ap.add_argument("--model", choices=("chronos", "moirai"))
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    ap.add_argument("--checkpoint-audit", type=Path, default=Path("artifacts/checkpoints/hf_audit.json"))
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--device", choices=("cuda",), default="cuda")
    args = ap.parse_args()
    if args.mode == "run":
        if not args.model:
            ap.error("--model required in run mode")
        run(args)
    else:
        if args.model:
            ap.error("--model must be omitted in audit mode")
        audit_pilot(args)


if __name__ == "__main__":
    main()
