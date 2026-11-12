"""TASK-004B.1 planning and environment preflight; NEVER runs inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

from cita2027.forecasting.full_plan import (
    make_full_plan, preflight_disk, publish_plan, validate_full_plan,
)
from cita2027.forecasting.cache import digest

# Previously accepted immutable identities: changes require a separate audit.
EXPECTED_PROCESSED_SHA = "4375f3b90c5dbfe1cfd1a8191a36c15c2d878993ef04b130c8de8810927103ba"
EXPECTED_SMOKE_CONFIG_SHA = "a2047e99ee7447b8a7194064f7d4ba627288b23527255d5b770a9bb217aad0cf"
EXPECTED_CHECKPOINT_AUDIT_SHA = "e426149cff7b3164793631276416ec10eb50b23b29048fb4b32e24ca35cbc974"
EXPECTED_MODELS = {
    "chronos": ("a971ba21945c4f1796b17a91fe69214b5f4ad472", "9c8b6fde5300f72b01c173153bf9288fa0a200614275bf0585071ad71a6a3d43"),
    "moirai": ("f5bd5d01f0d67de856107d43abdd637380aae0a3", "eaafeb166a3dfcf5728ab52a9ccc5392a23333b42a41a14e5528089ece3ef377"),
}
from cita2027.utils.provenance import sha256_file, canonical_json_bytes
from pilot_forecast_cache import identities, sources, verify_local_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("plan", "preflight", "verify-plan"))
    parser.add_argument("--model", choices=("chronos", "moirai"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument("--checkpoint-audit", type=Path, default=Path("artifacts/checkpoints/hf_audit.json"))
    args = parser.parse_args()
    if (args.mode == "preflight") != (args.model is not None):
        parser.error("--model is required only for --mode preflight")
    root = args.root.resolve()
    config = (root / args.config).resolve()
    audit = (root / args.checkpoint_audit).resolve()
    conf, df, source, checkpoints = sources(root, config, audit)
    if (source["processed_sha256"] != EXPECTED_PROCESSED_SHA
            or sha256_file(config) != EXPECTED_SMOKE_CONFIG_SHA
            or sha256_file(audit) != EXPECTED_CHECKPOINT_AUDIT_SHA):
        raise ValueError("Accepted SPY/config/checkpoint-audit identity drift")
    for backbone, (expected_revision, expected_weight) in EXPECTED_MODELS.items():
        _, _, revision, files = identities(conf, checkpoints, backbone)
        if revision != expected_revision or next(iter(files.values())) != expected_weight:
            raise ValueError(f"Accepted checkpoint identity drift: {backbone}")
    plan = make_full_plan(df, snapshot_id=conf["source_snapshot_id"],
        processed_sha256=source["processed_sha256"],
        smoke_config_sha256=sha256_file(config),
        checkpoint_audit_sha256=sha256_file(audit),
        seed=int(conf["seed"]), replay_origins=int(conf["replay_origins"]),
        replay_tolerance=float(conf["replay_tolerance"]))
    validate_full_plan(plan, df)
    plan_sha = digest(plan)
    path = root / "artifacts" / "forecast_cache" / "full_development" / "plans" / f"{plan_sha}.json"
    if args.mode == "plan":
        path, actual_sha = publish_plan(root, plan)
        if actual_sha != plan_sha:
            raise ValueError("Plan SHA changed")
    elif not path.is_file() or path.read_bytes() != canonical_json_bytes(plan):
        raise ValueError("Plan not published or plan bytes differ; run --mode plan first")
    result = {
        "task": "TASK-004B.1", "mode": args.mode,
        "status": "FULL_DEVELOPMENT_PLAN_VERIFIED_NOT_GPU_CLEARED",
        "plan_sha256": plan_sha, "plan_path": str(path),
        "count": plan["count"], "first": plan["first"], "last": plan["last"],
        "first_context_uses_warmup": df.iloc[plan["first"]["context_first_index"]]["split"] == "warmup",
        "validation_or_final_targets": 0, "pilot_forecasts_reused": False,
        "processed_sha256": plan["processed_sha256"],
        "contamination_free_verified": False,
        "point_in_time_adjusted_prices_verified": False,
    }
    if args.mode == "preflight":
        import torch
        from importlib.metadata import version, PackageNotFoundError
        model = args.model
        cls, model_id, revision, files = identities(conf, checkpoints, model)
        weight_sha = verify_local_checkpoint(model_id, revision, files)
        torch.cuda.init()
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
            raise ValueError("CUDA sm_120 mismatch")
        test = torch.ones((4, 4), device="cuda") @ torch.ones((4, 4), device="cuda")
        torch.cuda.synchronize()
        if not torch.isfinite(test).all().item():
            raise ValueError("CUDA arithmetic failed")
        required = "chronos-forecasting" if model == "chronos" else "uni2ts"
        version(required)
        disk = preflight_disk(root)
        if disk["status"] != "PASS":
            raise ValueError("Insufficient local free space")
        run_path = root / "artifacts" / "forecast_cache" / "full_development" / "runs" / plan_sha / model
        if run_path.exists() and any(run_path.iterdir()):
            raise ValueError("Nonempty full-development output path: inspect before launch")
        result["preflight"] = {
            "status": "PASS_FOR_PLANNING_ONLY", "model": model,
            "model_id": model_id, "model_revision": revision,
            "weight_sha256": weight_sha, "python": platform.python_version(),
            "python_executable": sys.executable, "torch": torch.__version__,
            "cuda_compiled": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "framework_version": version(required), "disk": disk,
            "run_path": str(run_path),
            "scientific_clearance": False,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
