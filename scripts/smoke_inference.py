"""Run a small, frozen, one-step return forecast audit for one backbone."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import yaml

from cita2027.forecasting.adapters import (
    ChronosT5Adapter, Moirai10Adapter, release_adapter, seed_inference,
)
from cita2027.forecasting.protocol import (
    CONTEXT_LENGTH, QUANTILE_LEVELS, choose_origins, load_processed,
    make_window, validate_smoke_records,
)
from cita2027.utils.provenance import sha256_file, write_json_atomic


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen SPY return forecast smoke test")
    ap.add_argument("--model", choices=["chronos", "moirai"], required=True)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    ap.add_argument("--checkpoint-audit", type=Path, required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()
    conf = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (conf["context_length"], conf["horizon"], conf["quantiles"]) != (256, 1, [0.1, 0.5, 0.9]):
        raise ValueError("Unexpected smoke protocol; do not silently change horizon/quantiles")
    if conf["samples"] < 10 or conf["replay_origins"] < 1 or conf["replay_origins"] > conf["count"]:
        raise ValueError("Invalid sampling/replay configuration")
    model_id = conf["model_ids"][args.model]
    approved_id = {"chronos": ChronosT5Adapter.checkpoint_id,
                   "moirai": Moirai10Adapter.checkpoint_id}[args.model]
    if model_id != approved_id:
        raise ValueError("Checkpoint ID differs from audited legacy model")
    checkpoint_list = json.loads(args.checkpoint_audit.read_text(encoding="utf-8"))
    matched = [r for r in checkpoint_list if r["model_id"] == model_id]
    if len(matched) != 1 or len(matched[0]["commit_sha"]) != 40:
        raise ValueError("Missing immutable checkpoint revision in audit")
    revision = matched[0]["commit_sha"]

    sid = conf["source_snapshot_id"]
    processed = args.root / "artifacts" / "processed" / f"{sid}.csv"
    source_manifest = args.root / "artifacts" / "manifests" / f"{sid}.json"
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source["source_id"] != sid or sha256_file(processed) != source["processed_sha256"]:
        raise ValueError("Processed artifact does not match Phase 2 manifest")
    df = load_processed(str(processed))
    origins = choose_origins(df, count=conf["count"])

    import torch
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run audit_gpu.py before inference")
        if torch.cuda.get_device_capability(0) != (12, 0):
            raise RuntimeError("Expected RTX Blackwell sm_120, different architecture detected")
    seed = int(conf["seed"])
    seed_inference(seed)
    adapter = (ChronosT5Adapter if args.model == "chronos" else Moirai10Adapter)(
        revision=revision, device=args.device, samples=int(conf["samples"])
    )

    # Hash local model weight files and compare against Hub LFS metadata.
    from huggingface_hub import hf_hub_download
    local_weights = {}
    for name, meta in matched[0].get("files", {}).items():
        if not name.endswith((".safetensors", ".bin")):
            continue
        local_path = hf_hub_download(repo_id=model_id, filename=name,
                                     revision=revision, local_files_only=True)
        actual_hash = sha256_file(Path(local_path))
        expected_hash = meta.get("lfs_sha256")
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(f"Checkpoint weight SHA-256 mismatch: {name}")
        local_weights[name] = {"sha256": actual_hash,
                               "verified_against_hub_lfs": bool(expected_hash)}
    if not local_weights:
        raise RuntimeError("No locally cached checkpoint weight files found to hash")

    records = []
    replay_max_diffs = []
    for i, origin in enumerate(origins):
        window = make_window(df, origin)
        # Reset RNG on each origin so a changed earlier forecast cannot alter later seeds.
        this_seed = seed + origin
        seed_inference(this_seed)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        q = adapter.predict(window.context)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0.0
        replay_diff = 0.0
        if i < conf["replay_origins"]:
            seed_inference(this_seed)
            replay_q = adapter.predict(window.context)
            replay_diff = float(np.max(np.abs(q - replay_q)))
            replay_max_diffs.append(replay_diff)
            if not np.allclose(q, replay_q, atol=float(conf["replay_tolerance"]), rtol=0):
                raise RuntimeError(f"Replay mismatch at origin={origin}: {replay_diff}")
        records.append({
            "model": args.model, "checkpoint_id": model_id, "checkpoint_commit": revision,
            "origin_index": origin, "target_index": origin + 1,
            "origin_date": window.origin_date, "target_date": window.target_date,
            "context_first_index": origin - CONTEXT_LENGTH + 1,
            "context_last_index": origin, "context_sha256": window.context_sha256,
            "target_log_return": window.target_log_return,
            "q10": float(q[0]), "q50": float(q[1]), "q90": float(q[2]),
            "elapsed_seconds": elapsed, "peak_allocated_mib": peak,
            "replay_max_abs_diff": replay_diff,
        })
        print(f"{args.model} origin {i+1}/{len(origins)} {window.origin_date}: ",
              f"q={q.tolist()} elapsed={elapsed:.3f}s peak={peak:.1f}MiB", flush=True)
    validate_smoke_records(records, expected_count=conf["count"])
    release_adapter(adapter)
    del adapter
    gc.collect()

    # Create a new immutable smoke run. No full forecast cache is created.
    run_id = f"{sid}_{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    out_dir = args.root / "artifacts" / "smoke" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    pred_path = out_dir / "predictions.csv"
    pd.DataFrame.from_records(records).to_csv(pred_path, index=False, float_format="%.17g")
    manifest = {
        "run_id": run_id, "model_id": model_id, "model_revision": revision,
        "model_files": matched[0].get("files", {}),
        "local_weight_file_hashes": local_weights,
        "processed_sha256": source["processed_sha256"], "processed_snapshot_id": sid,
        "smoke_config_sha256": sha256_file(args.config),
        "checkpoint_audit_sha256": sha256_file(args.checkpoint_audit),
        "predictions_sha256": sha256_file(pred_path),
        "context_length": 256, "horizon": 1, "target": "daily_log_return_decimal",
        "quantile_levels": list(QUANTILE_LEVELS),
        "samples": conf["samples"], "seed": seed, "count": len(records),
        "replay_max_abs_diff": max(replay_max_diffs),
        "sum_inference_seconds_excluding_replay_and_loading": sum(r["elapsed_seconds"] for r in records),
        "max_peak_allocated_mib_excluding_model_loading": max(r["peak_allocated_mib"] for r in records),
        "packages": {p: package_version(p) for p in (
            "torch", "chronos-forecasting", "uni2ts", "gluonts", "transformers",
            "huggingface-hub", "numpy", "pandas")},
        "python": platform.python_version(), "device": args.device,
        "cuda_compiled": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if args.device == "cuda" else None,
        "contamination_free_verified": False,
        "status": "SMOKE_PASS_NOT_FINAL_FORECAST_CACHE",
    }
    write_json_atomic(out_dir / "manifest.json", manifest)
    print(json.dumps({"status": "PASS", "predictions": str(pred_path),
                      "manifest": str(out_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
