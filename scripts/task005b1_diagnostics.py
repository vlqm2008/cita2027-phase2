"""TASK-005B.1: one-command, CPU-only diagnostics over immutable development outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from cita2027.calibration.diagnostic_correction import (
    PLAN_SHA, REPORT_SHA, load_traces, evaluate_report, validate_input_report,
)
from cita2027.forecasting.cache import store_or_verify, validate_cross_model
from cita2027.forecasting.protocol import make_window
from cita2027.utils.provenance import sha256_file
from full_development_cache import MODELS, EXPECTED_PLAN_SHA, prepare, identity, paths, verify_complete


def run(root: Path, config: Path, audit: Path) -> dict:
    if EXPECTED_PLAN_SHA != PLAN_SHA:
        raise ValueError('Accepted full plan SHA in runner changed')
    conf, df, checkpoints, plan, view, plan_sha = prepare(root, config, audit)
    if plan_sha != PLAN_SHA or len(plan['origins']) != 1258:
        raise ValueError('Accepted full development plan drift')
    base = root / 'artifacts' / 'calibration' / 'task005b' / PLAN_SHA
    report_path = base / 'full_development_W252_expansion_only.json'
    report = validate_input_report(report_path)
    trace_path = root / report['diagnostic_trace_path']
    traces = load_traces(trace_path, report)
    records = {}
    for model in MODELS:
        _, model_id, revision, weight, _ = identity(conf, checkpoints, model)
        # Use the original config for model identity (do not derive it from trace).
        # identity may inspect model metadata; prepare has already checked the config.
        run_dir = paths(root, plan_sha, model)
        manifest, rows = verify_complete(run_dir, view, plan_sha, model,
                                         model_id, revision, weight, df)
        if sha256_file(run_dir / 'manifest.json') != report['inputs'][model]['manifest_sha256']:
            raise ValueError(f'{model} manifest differs from accepted TASK-005B')
        if sha256_file(run_dir / 'predictions.csv') != report['inputs'][model]['predictions_sha256']:
            raise ValueError(f'{model} forecast CSV differs from accepted TASK-005B')
        records[model] = rows
    validate_cross_model(records['chronos'], records['moirai'])
    context = make_window(df, plan['origins'][0]['origin_index']).context
    result = evaluate_report(report, traces, records, context)
    result['diagnostic_trace_sha256'] = sha256_file(trace_path)
    result['inputs'] = report['inputs']
    result['base_report_sha256'] = REPORT_SHA
    path = root / 'artifacts' / 'calibration' / 'task005b1' / PLAN_SHA / 'diagnostic_correction_v1.json'
    store_or_verify(path, result)
    brief = {'status': result['status'], 'regime_counts': result['regime_counts'],
             'report_path': str(path), 'report_sha256': sha256_file(path),
             'models': {m: {
                 'gamma_upper_fraction': result['models'][m]['gamma']['none__variable_gamma_aci']['driver_at_max_fraction'],
                 'fixed_0_02_winkler': {s: result['models'][m]['fixed_gamma_0_02_sensitivity'][f'{s}__metrics']['winkler_mean']
                                      for s in ('none', 'residual_ewma', 'return_ewma')},
                 'variable_minus_fixed_CI20': {
                     s: result['models'][m]['direct_paired_contrasts'][f'{s}__variable_gamma_aci_minus_{s}__fixed_gamma_aci']['20']['ci_95_percentile']
                     for s in ('none', 'residual_ewma', 'return_ewma')},
             } for m in MODELS}, 'scientific_clearance': False}
    return brief


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--config', type=Path, default=Path('configs/smoke.yaml'))
    p.add_argument('--checkpoint-audit', type=Path, default=Path('artifacts/checkpoints/hf_audit.json'))
    args = p.parse_args()
    root = args.root.resolve()
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_task005b1_diagnostics.py'],
                   cwd=root, check=True)
    outcome = run(root, root / args.config, root / args.checkpoint_audit)
    print(json.dumps(outcome, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
