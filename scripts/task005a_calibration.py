"""TASK-005A: one-command CPU tests + read-only forecast-cache audit + pilot calibration.

Never runs TSFM inference. Never accesses validation or final test targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from cita2027.calibration.online import (CONTROLLERS, SCALES, Parameters,
                                         calibrate, interval_metrics)
from cita2027.forecasting.cache import store_or_verify, validate_cross_model
from cita2027.forecasting.protocol import make_window
from cita2027.utils.provenance import sha256_file
from full_development_cache import (EXPECTED_PLAN_SHA, MODELS, identity, paths,
                                    prepare, verify_complete)


def analyze(root: Path, config: Path, audit: Path, pilot_length: int, params: Parameters):
    conf, df, checkpoints, plan, view, sha = prepare(root, config, audit)
    if sha != EXPECTED_PLAN_SHA or plan["count"] != 1258:
        raise ValueError("Accepted full-development plan changed")
    if not (params.window + 1 <= pilot_length <= plan["count"]):
        raise ValueError("Pilot must leave at least one evaluated origin and stay in development")
    input_rows = {}
    input_manifest_hashes = {}
    for model in MODELS:
        _, model_id, rev, weight, _ = identity(conf, checkpoints, model)
        model_dir = paths(root, sha, model)
        manifest, records = verify_complete(model_dir, view, sha, model, model_id, rev, weight, df)
        if len(records) != 1258:
            raise ValueError("Incomplete forecast cache")
        input_rows[model] = records
        input_manifest_hashes[model] = sha256_file(model_dir / "manifest.json")
        if (manifest["contamination_free_verified"] is not False or
                manifest["point_in_time_adjusted_prices_verified"] is not False):
            raise ValueError("Invalid scientific risk flags")
    validate_cross_model(input_rows["chronos"], input_rows["moirai"])
    first_origin = plan["origins"][0]["origin_index"]
    first_context = make_window(df, first_origin).context
    outputs = {}
    for model in MODELS:
        outputs[model] = {}
        for scale in SCALES:
            for controller in CONTROLLERS:
                rows = calibrate(input_rows[model], first_context, params, scale,
                                 controller, pilot_length)
                metrics = interval_metrics(rows, params.alpha)
                # Descriptive, ex-ante volatility stratification; NOT conditional coverage.
                ref = max(float(np.sqrt(np.mean(first_context**2))), params.min_scale)
                high = [r for r in rows if r["return_volatility"] > ref]
                low = [r for r in rows if r["return_volatility"] <= ref]
                metrics["regime_empirical_coverage"] = {
                    "threshold_source": "first_origin_context_rms",
                    "high_vol": {"n": len(high), "coverage": interval_metrics(high, params.alpha)["coverage"] if high else None},
                    "other": {"n": len(low), "coverage": interval_metrics(low, params.alpha)["coverage"] if low else None},
                    "is_conditional_coverage_guarantee": False,
                }
                outputs[model][f"{scale}__{controller}"] = {
                    "metrics": metrics, "rows": rows,
                }
    report = {
        "task": "TASK-005A", "status": "CAUSAL_CALIBRATION_TECHNICAL_PILOT_NOT_COMPARATIVE_EVIDENCE",
        "full_plan_sha256": sha, "processed_sha256": plan["processed_sha256"],
        "input_manifest_sha256": input_manifest_hashes,
        "protocol": {"nominal_miscoverage": params.alpha, "window": params.window,
                     "ewma_decay": params.decay, "gamma_base": params.gamma,
                     "gamma_driver": "same_return_ewma_ratio_all_scaling_conditions",
                     "gamma_driver_bounds": [params.driver_min, params.driver_max],
                     "alpha_clip": [1/(params.window+1), 1-1/(params.window+1)],
                     "correction": params.correction, "initialization": "first_origin_256_return_rms",
                     "burn_in_origins": params.window, "pilot_length": pilot_length,
                     "evaluation_origins": pilot_length - params.window,
                     "hyperparameters": "fixed_illustrative_not_tuned"},
        "models": outputs, "contamination_free_verified": False,
        "point_in_time_adjusted_prices_verified": False, "scientific_clearance": False,
        "validation_used": False, "final_test_used": False,
    }
    out_dir = root / "artifacts" / "calibration" / "task005a" / sha
    out_path = out_dir / f"pilot_{pilot_length}_W{params.window}_{params.correction}.json"
    store_or_verify(out_path, report)
    summary = {model: {cell: result["metrics"] for cell, result in items.items()}
               for model, items in outputs.items()}
    return {"status": "TASK005A_TECHNICAL_PILOT_PASS_NOT_SCIENTIFIC_CLEARANCE",
            "models": list(MODELS), "factorial_cells_per_model": 9,
            "pilot_origins": pilot_length, "burn_in": params.window,
            "evaluated_origins_per_cell": pilot_length - params.window,
            "full_plan_sha256": sha, "report_path": str(out_path),
            "report_sha256": sha256_file(out_path),
            "summary": summary, "scientific_clearance": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    ap.add_argument("--checkpoint-audit", type=Path, default=Path("artifacts/checkpoints/hf_audit.json"))
    ap.add_argument("--pilot-length", type=int, default=320)
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    if not args.skip_tests:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_task005a_calibration.py"], cwd=root)
        if p.returncode:
            raise SystemExit("TASK-005A tests FAILED; pilot not run")
    result = analyze(root, root / args.config, root / args.checkpoint_audit,
                     args.pilot_length, Parameters())
    # Print compact results without dumping all 18 x 68 per-origin records.
    result["summary"] = {m: {cell: {k: round(v, 8) if isinstance(v, float) else v
                 for k, v in v0.items() if k in ("winkler_mean", "coverage", "width_mean",
                     "lower_miss_rate", "upper_miss_rate", "n")}
                 for cell, v0 in items.items()} for m, items in result["summary"].items()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
