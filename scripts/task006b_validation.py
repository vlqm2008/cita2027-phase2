"""TASK-006B: frozen SPY Validation execution orchestration and immutable audit.

The accepted TASK-006A worker performs model-specific GPU inference, write-once
records and causal calibration replay. This wrapper adds an *exact plan-hash*
gate, focused tests, a single command, and a comparative scientific summary.
It never fine-tunes or selects methods from Validation results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import task006a_validation as accepted
from cita2027.calibration.online import SCALES, CONTROLLERS
from cita2027.forecasting.cache import digest, store_or_verify
from cita2027.utils.provenance import canonical_json_bytes, sha256_file

LOCKED_VALIDATION_SHA = 'f574fd5be7e6537c263aa8c9a646dbb8676d87a25b38a70c83131cb0e894fe54'
DEV_PLAN_SHA = accepted.DEV_PLAN_SHA
SENSITIVE_FLAGS = ('contamination_free_verified', 'point_in_time_adjusted_prices_verified', 'scientific_clearance')
CELLS = tuple(f'{s}__{c}' for s in SCALES for c in CONTROLLERS)
EXPECTED_COUNT = 502


def verify_locked_plan(root: Path, plan: dict, expected_sha: str = LOCKED_VALIDATION_SHA) -> Path:
    """Reject changed or unpublished plans *before* any GPU process is launched."""
    sha = digest(plan)
    path = root / 'artifacts' / 'forecast_cache' / 'validation' / 'plans' / f'{expected_sha}.json'
    if expected_sha != LOCKED_VALIDATION_SHA or sha != expected_sha or not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError('BLOCKER: immutable Validation plan identity mismatch')
    if path.read_bytes() != canonical_json_bytes(plan):
        raise ValueError('BLOCKER: published Validation plan differs from reconstructed plan')
    if (plan['development_plan_sha256'] != DEV_PLAN_SHA or plan['count'] != EXPECTED_COUNT
            or plan['bridge_count'] != 1 or plan['first']['origin_index'] != 4530
            or plan['first']['target_index'] != 4531 or plan['last']['target_index'] != 5032
            or plan['factorial_scales'] != list(SCALES)
            or plan['factorial_controllers'] != list(CONTROLLERS)
            or plan['nominal_miscoverage'] != 0.2 or plan['calibration_window'] != 252
            or plan['base_gamma'] != 0.01 or plan['ewma_decay'] != 0.94
            or plan['shared_gamma_driver'] != 'return_ewma'
            or plan['gamma_driver_bounds'] != [0.5, 2.0]
            or plan['bootstrap_lengths'] != [10, 20, 40]
            or plan['primary_correction'] != 'expansion_only'
            or any(plan.get(flag) is not False for flag in SENSITIVE_FLAGS)):
        raise ValueError('BLOCKER: published Validation protocol drift')
    origins = plan['origins']
    if (len(origins) != EXPECTED_COUNT or
            [r['origin_index'] for r in origins] != list(range(4530, 5032)) or
            any(r['target_index'] != r['origin_index'] + 1 or
                r['seed'] != 2027 + r['origin_index'] for r in origins)):
        raise ValueError('BLOCKER: Validation origin/target/seed alignment drift')
    return path



def verify_development_anchors(root: Path) -> None:
    """O(1) hash checks of accepted CPU calibration artifacts; no reruns."""
    path = root / 'artifacts' / 'calibration' / 'task005b' / DEV_PLAN_SHA / 'full_development_W252_expansion_only.json'
    if not path.is_file() or sha256_file(path) != accepted.ACCEPTED_DEV_REPORT_SHA:
        raise ValueError('BLOCKER: accepted Development calibration report drift')
    report = json.loads(path.read_text(encoding='utf-8'))
    if (report.get('full_plan_sha256') != DEV_PLAN_SHA or
            report.get('diagnostic_trace_sha256') != accepted.ACCEPTED_DEV_TRACE_SHA or
            any(report.get(flag) is not False for flag in SENSITIVE_FLAGS)):
        raise ValueError('BLOCKER: Development calibration report identity drift')
    trace = root / report['diagnostic_trace_path']
    if not trace.is_file() or sha256_file(trace) != accepted.ACCEPTED_DEV_TRACE_SHA:
        raise ValueError('BLOCKER: accepted Development per-origin trace drift')


def verify_report(report: dict, plan: dict) -> None:
    """Check report structure, sample identity and scientific safety flags."""
    if (report.get('validation_plan_sha256') != digest(plan)
            or report.get('development_plan_sha256') != DEV_PLAN_SHA
            or report.get('records_per_model') != EXPECTED_COUNT
            or report.get('final_test_used') is not False
            or report.get('method_selection_performed') is not False
            or report.get('validation_hyperparameter_tuning') is not False
            or any(report.get(flag) is not False for flag in SENSITIVE_FLAGS)):
        raise ValueError('BLOCKER: unsafe Validation report metadata')
    counts = report.get('regime_counts', {})
    if (counts.get('high', 0) + counts.get('other', 0) != EXPECTED_COUNT or
            min(counts.get('high', 0), counts.get('other', 0)) < 1):
        raise ValueError('BLOCKER: empty or invalid Validation volatility regime')
    if set(report.get('models', {})) != set(accepted.MODELS):
        raise ValueError('BLOCKER: missing/extra backbone results')
    for model in accepted.MODELS:
        methods = report['models'][model]
        if not set(CELLS).issubset(methods):
            raise ValueError(f'BLOCKER: factorial incompleteness: {model}')
        for cell in CELLS:
            item = methods[cell]
            metrics = item['metrics']
            if (metrics['n'] != EXPECTED_COUNT or
                    not np.isfinite(metrics['winkler_mean']) or
                    not np.isfinite(metrics['coverage']) or
                    not 0 <= metrics['coverage'] <= 1 or
                    not np.isfinite(metrics['width_mean']) or metrics['width_mean'] < 0 or
                    not np.isclose(metrics['lower_miss_rate'] +
                                   metrics['upper_miss_rate'] + metrics['coverage'], 1, atol=1e-12)):
                raise ValueError(f'BLOCKER: invalid metrics: {model}/{cell}')
            for regime in ('high', 'other'):
                if item['regimes'][regime]['n'] != counts[regime]:
                    raise ValueError(f'BLOCKER: regime sample misalignment: {model}/{cell}')
                if item['regimes'][regime]['valid_bootstrap_replicates'] < 1:
                    raise ValueError(f'BLOCKER: no valid regime bootstrap replicates: {model}/{cell}')
            for comparator in ('unscaled_rolling', 'same_scale_fixed'):
                comparisons = item['paired_contrasts'][comparator]
                if set(comparisons) != {'10', '20', '40'}:
                    raise ValueError('BLOCKER: missing bootstrap length sensitivity')
                for L in ('10', '20', '40'):
                    d = comparisons[L]
                    if (not np.isfinite(d['mean']) or len(d['ci95']) != 2 or
                            not np.isfinite(d['ci95']).all() or d['ci95'][0] > d['ci95'][1] or
                            d['exploratory'] is not True or d['multiplicity_adjusted'] is not False):
                        raise ValueError('BLOCKER: invalid exploratory paired CI')
        if methods['raw_q10_q90']['metrics']['n'] != EXPECTED_COUNT:
            raise ValueError('BLOCKER: raw baseline not aligned')
        signed = methods['signed_rolling_none']
        if signed['status'] == 'FINITE_SIGNED_INTERVALS':
            if signed['metrics']['n'] != EXPECTED_COUNT or signed['empty_origins_in_validation']:
                raise ValueError('BLOCKER: signed-CQR reference mismatch')
        elif signed['status'] == 'VALIDATION_EMPTY_SIGNED_SET':
            if not signed['empty_origins_in_validation'] or signed['metrics'] is not None:
                raise ValueError('BLOCKER: invalid signed-CQR empty-set accounting')
        else:
            raise ValueError('BLOCKER: unrecognized signed-CQR status')


def summarize(report: dict, dev: dict, plan: dict, validation_report_sha: str,
              forecast_provenance: dict) -> dict:
    """Descriptive development/validation differences, with no method selection."""
    verify_report(report, plan)
    if dev.get('full_plan_sha256') != DEV_PLAN_SHA:
        raise ValueError('Accepted Development summary origin plan drift')
    models = {}
    for model in accepted.MODELS:
        comparisons = {}
        for cell in CELLS:
            dev_m = dev['models'][model]['cells'][cell]['metrics']
            val_m = report['models'][model][cell]['metrics']
            comparisons[cell] = {
                'development_n': int(dev_m['n']),
                'validation_n': int(val_m['n']),
                'development_winkler': float(dev_m['winkler_mean']),
                'validation_winkler': float(val_m['winkler_mean']),
                'validation_minus_development_winkler': float(val_m['winkler_mean'] - dev_m['winkler_mean']),
                'development_coverage': float(dev_m['coverage']),
                'validation_coverage': float(val_m['coverage']),
                'validation_width': float(val_m['width_mean']),
                'validation_lower_miss_rate': float(val_m['lower_miss_rate']),
                'validation_upper_miss_rate': float(val_m['upper_miss_rate']),
            }
        raw_dev = dev['models'][model]['native_q10_q90']['metrics']
        raw_val = report['models'][model]['raw_q10_q90']['metrics']
        models[model] = {
            'all_nine_factorial_cells': comparisons,
            'raw_baseline_development_winkler': float(raw_dev['winkler_mean']),
            'raw_baseline_validation_winkler': float(raw_val['winkler_mean']),
            'raw_baseline_development_coverage': float(raw_dev['coverage']),
            'raw_baseline_validation_coverage': float(raw_val['coverage']),
            'signed_cqr_status': report['models'][model]['signed_rolling_none']['status'],
            'forecast_artifact_provenance': forecast_provenance[model],
        }
    return {
        'status': 'VALIDATION_REPORT_AUDIT_PASS_NOT_SCIENTIFIC_CLEARANCE',
        'task': 'TASK-006B',
        'validation_plan_sha256': digest(plan),
        'development_plan_sha256': DEV_PLAN_SHA,
        'development_report_sha256': accepted.ACCEPTED_DEV_REPORT_SHA,
        'validation_report_sha256': validation_report_sha,
        'models': models,
        'regime_counts': report['regime_counts'],
        'paired_analysis': 'in_validation_calibration_report_json_blocks_10_20_40',
        'regime_cis': 'in_validation_calibration_report_json_block_20',
        'development_validation_comparison': 'descriptive_not_a_paired_test_across_periods',
        'selection_performed': False,
        'statistical_scope': 'exploratory_no_multiplicity_adjustment',
        'scientific_interpretation': {
            'VERIFIED': 'Technical integrity, source alignment, continuous calibration replay, and all-cell metrics when validation audit passes.',
            'EXPLORATORY': 'Paired moving-block confidence intervals for prespecified contrasts; no multiple-comparison correction.',
            'UNKNOWN': 'Out-of-training contamination status, point-in-time price revisions, and scientific validity beyond the measured sample.'
        },
        'final_test_used': False,
        'contamination_free_verified': False,
        'point_in_time_adjusted_prices_verified': False,
        'scientific_clearance': False,
    }


def audit(root: Path, expected_sha: str = LOCKED_VALIDATION_SHA) -> dict:
    _, _, _, _, _, plan = accepted.inputs(root)
    verify_locked_plan(root, plan, expected_sha)
    # This is the accepted TASK-006A CPU audit; it includes the causal prefix
    # continuity and cross-backbone alignment check, without rerunning inference.
    execution = accepted.evaluate(root)
    if execution['status'] != 'VALIDATION_CALIBRATION_AUDIT_PASS_NOT_SCIENTIFIC_CLEARANCE':
        raise ValueError('Validation calibration audit did not pass')
    report_path = Path(execution['report_path'])
    val_sha = sha256_file(report_path)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report_path.read_bytes() != canonical_json_bytes(report):
        raise ValueError('Validation report noncanonical')
    if sha256_file(root / 'artifacts' / 'calibration' / 'validation' / expected_sha / 'validation_502_per_cell.csv') != report['trace_sha256']:
        raise ValueError('Validation calibration trace hash drift')
    dev_path = root / 'artifacts' / 'calibration' / 'task005b' / DEV_PLAN_SHA / 'full_development_W252_expansion_only.json'
    if sha256_file(dev_path) != accepted.ACCEPTED_DEV_REPORT_SHA:
        raise ValueError('Development report drift')
    dev = json.loads(dev_path.read_text(encoding='utf-8'))
    comparison = summarize(report, dev, plan, val_sha, report['forecast_inputs'])
    summary_path = root / 'artifacts' / 'calibration' / 'validation' / expected_sha / 'task006b_scientific_comparison.json'
    store_or_verify(summary_path, comparison)
    return {'status': comparison['status'], 'validation_plan_sha256': expected_sha,
            'forecast_count_per_model': EXPECTED_COUNT, 'total_forecasts': 1004,
            'validation_report_path': str(report_path), 'validation_report_sha256': val_sha,
            'comparison_path': str(summary_path), 'comparison_sha256': sha256_file(summary_path),
            'regime_counts': report['regime_counts'],
            'scientific_clearance': False,
            'final_test_used': False}


def run(root: Path):
    _, _, _, _, _, plan = accepted.inputs(root)
    verify_locked_plan(root, plan)  # before tests, writes, or GPU
    verify_development_anchors(root)
    tests = root / 'tests' / 'test_task006b_validation.py'
    subprocess.run([sys.executable, '-m', 'pytest', '-q', str(tests)], cwd=root, check=True)
    # Never launch both models simultaneously: 8GB VRAM requires sequential loads.
    for model in accepted.MODELS:
        interpreter = root / ('.venv' if model == 'chronos' else '.venv-moirai') / 'Scripts' / 'python.exe'
        if not interpreter.is_file():
            raise FileNotFoundError(f'Missing accepted model environment: {interpreter}')
        command = [str(interpreter), str(Path(accepted.__file__).resolve()), '_worker',
                   '--root', str(root), '--model', model, '--allow-validation']
        subprocess.run(command, cwd=root, env={**os.environ, 'PYTHONPATH': str(root / 'src')}, check=True)
    return audit(root)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['preflight', 'run', 'audit'])
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--allow-validation', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    if a.command == 'preflight':
        _, _, _, _, _, plan = accepted.inputs(root)
        file = verify_locked_plan(root, plan)
        verify_development_anchors(root)
        output = {'status': 'TASK006B_PREFLIGHT_READY_NO_GPU',
                  'plan_path': str(file), 'plan_sha256': LOCKED_VALIDATION_SHA,
                  'forecast_count_per_model': EXPECTED_COUNT,
                  'scientific_clearance': False}
    elif a.command == 'audit':
        output = audit(root)
    else:
        if not a.allow_validation:
            raise ValueError('GPU execution is opt-in: supply --allow-validation')
        output = run(root)
    print(json.dumps(output, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
