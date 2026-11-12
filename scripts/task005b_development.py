"""TASK-005B: one command for tests + immutable full-development calibration analysis.

CPU-only. Reads the accepted immutable forecast caches; never calls TSFM adapters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from cita2027.calibration.online import Parameters
from cita2027.calibration.development_analysis import (
    BLOCK_LENGTH, BOOTSTRAP_REPS, BOOTSTRAP_SEED, analyze_model,
    circular_block_indices,
)
from cita2027.forecasting.cache import store_or_verify, validate_cross_model, write_once_atomic
from cita2027.forecasting.protocol import make_window
from cita2027.utils.provenance import sha256_file
from full_development_cache import EXPECTED_PLAN_SHA, MODELS, identity, paths, prepare, verify_complete


def immutable_bytes(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'Immutable analysis artifact has changed: {path}')
    else:
        try:
            write_once_atomic(path, content)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f'Concurrent conflicting output: {path}') from None


def run_analysis(root: Path, config: Path, audit_path: Path, reps: int = BOOTSTRAP_REPS,
                 block: int = BLOCK_LENGTH, bootstrap_seed: int = BOOTSTRAP_SEED) -> dict:
    if (reps != BOOTSTRAP_REPS or block != BLOCK_LENGTH or bootstrap_seed != BOOTSTRAP_SEED):
        raise ValueError('Primary bootstrap specification locked; use separate sensitivity artifact')
    params = Parameters()
    conf, df, checkpoints, plan, view, plan_sha = prepare(root, config, audit_path)
    if plan_sha != EXPECTED_PLAN_SHA or len(plan['origins']) != 1258:
        raise ValueError('Accepted origin plan drift')
    rows_by_model, input_sha = {}, {}
    for model in MODELS:
        _, model_id, revision, weight, _ = identity(conf, checkpoints, model)
        run_dir = paths(root, plan_sha, model)
        manifest, records = verify_complete(run_dir, view, plan_sha, model,
                                            model_id, revision, weight, df)
        if (len(records) != 1258 or manifest['contamination_free_verified'] is not False
                or manifest['point_in_time_adjusted_prices_verified'] is not False):
            raise ValueError('Full raw cache completeness/provenance failure')
        rows_by_model[model] = records
        input_sha[model] = {'manifest_sha256': sha256_file(run_dir / 'manifest.json'),
                            'predictions_sha256': sha256_file(run_dir / 'predictions.csv')}
    validate_cross_model(rows_by_model['chronos'], rows_by_model['moirai'])
    first = make_window(df, plan['origins'][0]['origin_index']).context
    n_eval = len(plan['origins']) - params.window
    if n_eval != 1006:
        raise ValueError('Evaluation length drift')
    draws = circular_block_indices(n_eval, reps, block, bootstrap_seed)
    output = {}
    trace_frames = []
    for model in MODELS:
        model_result, trace_rows = analyze_model(rows_by_model[model], first, params, draws)
        output[model] = model_result
        for r in trace_rows:
            trace_frames.append({'model': model, **r})
    trace = pd.DataFrame.from_records(trace_frames)
    trace = trace.sort_values(['model', 'cell', 'origin_index'], kind='stable').reset_index(drop=True)
    trace_csv = trace.to_csv(index=False, float_format='%.17g').encode('utf-8')
    out_dir = root / 'artifacts' / 'calibration' / 'task005b' / plan_sha
    trace_path = out_dir / 'full_development_1006_per_cell.csv'
    immutable_bytes(trace_path, trace_csv)
    import hashlib
    trace_sha = hashlib.sha256(trace_csv).hexdigest()
    report = {
        'task': 'TASK-005B', 'status': 'FULL_DEVELOPMENT_CALIBRATION_FILE_AUDIT_PASS_NOT_SCIENTIFIC_CLEARANCE',
        'full_plan_sha256': plan_sha, 'processed_sha256': plan['processed_sha256'],
        'inputs': input_sha, 'protocol': {
            'window': params.window, 'nominal_miscoverage': params.alpha,
            'factorial_scaling': ['none', 'residual_ewma', 'return_ewma'],
            'factorial_controllers': ['rolling_cqr', 'fixed_gamma_aci', 'variable_gamma_aci'],
            'primary_correction': params.correction, 'quantile_method': 'conformal_order_statistic',
            'decay': params.decay, 'gamma': params.gamma, 'min_scale': params.min_scale,
            'driver_bounds': [params.driver_min, params.driver_max],
            'gamma_driver': 'shared_return_ewma',
            'alpha_clipping': [1 / (params.window + 1), 1 - 1 / (params.window + 1)],
            'burn_in_origins': params.window, 'evaluation_origins': n_eval,
            'bootstrap': {'method': 'paired_circular_moving_blocks', 'block_length': block,
                          'replicates': reps, 'seed': bootstrap_seed, 'ci': 'percentile_95',
                          'multiplicity_adjustment': None, 'exploratory': True},
            'hyperparameters': 'prespecified_task005a_pilot_constants_not_retuned',
        },
        'diagnostic_trace_path': str(trace_path.relative_to(root).as_posix()),
        'diagnostic_trace_sha256': trace_sha,
        'models': output,
        'validation_used': False, 'final_test_used': False,
        'contamination_free_verified': False,
        'point_in_time_adjusted_prices_verified': False, 'scientific_clearance': False,
    }
    report_path = out_dir / 'full_development_W252_expansion_only.json'
    store_or_verify(report_path, report)
    summary = {}
    for model in MODELS:
        cells = output[model]['cells']
        summary[model] = {
            'raw_q10_q90': {k: round(v, 8) if isinstance(v, float) else v
                           for k, v in output[model]['native_q10_q90']['metrics'].items()
                           if k in ('winkler_mean', 'coverage', 'width_mean')},
            'signed_rolling_none': {'status': output[model]['signed_rolling_none']['status'],
                                    'empty_count': output[model]['signed_rolling_none']['empty_count'],
                                    'winkler_mean': (output[model]['signed_rolling_none']['metrics'] or {}).get('winkler_mean')},
            'cells': {cell: {'winkler_mean': round(obj['metrics']['winkler_mean'], 8),
                             'coverage': round(obj['metrics']['coverage'], 8),
                             'width_mean': round(obj['metrics']['width_mean'], 8),
                             'lower_miss_rate': round(obj['metrics']['lower_miss_rate'], 8),
                             'upper_miss_rate': round(obj['metrics']['upper_miss_rate'], 8),
                             'paired_vs_raw_ci95': output[model]['paired_vs_native'][cell]['ci_95_percentile'],
                             'alpha_clip_events': (obj['diagnostics']['alpha_next_clip_lower_events']
                                                   + obj['diagnostics']['alpha_next_clip_upper_events'])}
                      for cell, obj in cells.items()},
        }
    return {'status': report['status'], 'development_origins': 1258,
            'evaluated_per_cell': n_eval, 'factorial_cells_per_model': 9,
            'report_path': str(report_path), 'report_sha256': sha256_file(report_path),
            'trace_sha256': trace_sha, 'summary': summary,
            'scientific_clearance': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--config', type=Path, default=Path('configs/smoke.yaml'))
    parser.add_argument('--checkpoint-audit', type=Path, default=Path('artifacts/checkpoints/hf_audit.json'))
    args = parser.parse_args()
    root = args.root.resolve()
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q',
                            'tests/test_task005a_calibration.py', 'tests/test_task005b_analysis.py'], cwd=root)
    if tests.returncode:
        raise SystemExit('TASK-005B CPU tests FAILED; analysis not run')
    result = run_analysis(root, root / args.config, root / args.checkpoint_audit)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
