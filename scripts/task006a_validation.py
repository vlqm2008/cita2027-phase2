"""TASK-006A: plan now; optional explicitly authorized Validation execution later.

`plan` never loads TSFMs or reads validation model outputs. `run` is opt-in.
Development cache and accepted calibrator modules remain unmodified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from cita2027.calibration.online import Parameters, SCALES, CONTROLLERS, calibrate, interval_metrics, signed_score, conformal_order_statistic
from cita2027.calibration.development_analysis import score_arrays, circular_block_indices
from cita2027.calibration.diagnostic_correction import causal_regime_indices
from cita2027.forecasting.cache import digest, store_or_verify, validate_cross_model, write_once_atomic
from cita2027.forecasting.protocol import check_quantiles, make_window
from cita2027.utils.provenance import canonical_json_bytes, sha256_file
from cita2027.validation.boundary import (DEV_PLAN_SHA, VAL_COUNT, make_validation_plan,
    publish_validation_plan, validation_window, check_boundary_records)
from full_development_cache import (MODELS, EXPECTED_PLAN_SHA, prepare, identity,
    paths as dev_paths, verify_complete, preflight_worker)
from pilot_forecast_cache import version, verify_local_checkpoint
from cita2027.forecasting.adapters import seed_inference

ACCEPTED_DEV_REPORT_SHA = '972a196f7beae4e31a343a1163beb678f76ee29e63362bb9b10171d3ea1ff149'
ACCEPTED_DEV_TRACE_SHA = '2f6f88345e5808404ddd609b535f64e3c6a64d0ab11c213305e0f9c73cf39e45'


def inputs(root: Path):
    config = root / 'configs' / 'smoke.yaml'
    audit = root / 'artifacts' / 'checkpoints' / 'hf_audit.json'
    conf, df, checkpoints, dev_plan, dev_view, sha = prepare(root, config, audit)
    if sha != DEV_PLAN_SHA or sha != EXPECTED_PLAN_SHA:
        raise ValueError('Development origin plan identity drift')
    val_plan = make_validation_plan(df, dev_plan, sha256_file(audit))
    return conf, df, checkpoints, dev_plan, dev_view, val_plan


def val_dir(root: Path, sha: str, model: str) -> Path:
    return root / 'artifacts' / 'forecast_cache' / 'validation' / 'runs' / sha / model


def record_validate(r: dict, expected: dict, plan: dict, sha: str, model: str,
                    model_id: str, rev: str, weight: str, df: pd.DataFrame) -> None:
    w = validation_window(df, expected['origin_index'])
    fields = {'schema_version': 'validation-forecast-v1', 'plan_sha256': sha,
              'processed_sha256': plan['processed_sha256'],
              'processed_snapshot_id': plan['source_snapshot_id'], 'model': model,
              'checkpoint_id': model_id, 'checkpoint_commit': rev,
              'checkpoint_weight_sha256': weight, 'origin_index': expected['origin_index'],
              'target_index': expected['target_index'], 'origin_date': w.origin_date,
              'target_date': w.target_date, 'context_sha256': w.context_sha256,
              'seed': expected['seed'], 'context_length': 256, 'horizon': 1,
              'samples': 100, 'quantile_levels': [0.1, 0.5, 0.9],
              'target_units': 'daily_log_return_decimal',
              'replay_checked': expected == plan['origins'][0] or expected == plan['origins'][1],
              'contamination_free_verified': False,
              'point_in_time_adjusted_prices_verified': False}
    if not all(r.get(k) == v for k, v in fields.items()):
        raise ValueError(f'Validation record identity/split mismatch {expected["origin_index"]}')
    if (r.get('context_first_index') != expected['context_first_index'] or
            r.get('context_last_index') != expected['context_last_index'] or
            float(r.get('target_log_return', float('nan'))).hex() != w.target_log_return.hex()):
        raise ValueError('Incorrect context or target alignment')
    check_quantiles([r['q10'], r['q50'], r['q90']])
    if not (np.isfinite(r['elapsed_seconds']) and r['elapsed_seconds'] >= 0 and
            np.isfinite(r['peak_allocated_mib']) and r['peak_allocated_mib'] >= 0 and
            0 <= r['replay_max_abs_diff'] <= plan['replay_tolerance']):
        raise ValueError('Invalid replay/runtime measurements')


def load_records(root, model, plan, df, model_id, rev, weight, allow_partial):
    sha = digest(plan)
    folder = val_dir(root, sha, model) / 'records'
    expected_names = {f'{r["origin_index"]:06d}.json' for r in plan['origins']}
    found = {f.name for f in folder.glob('*.json')} if folder.exists() else set()
    if found - expected_names:
        raise ValueError('Unexpected out-of-plan validation records')
    # Publish only an initial contiguous prefix; reject gaps to prevent cherry-picking.
    rows = []
    for row in plan['origins']:
        p = folder / f'{row["origin_index"]:06d}.json'
        if not p.exists():
            if not allow_partial:
                raise ValueError('Missing validation record')
            break
        r = json.loads(p.read_text(encoding='utf-8'))
        if p.read_bytes() != canonical_json_bytes(r):
            raise ValueError('Noncanonical validation record')
        record_validate(r, row, plan, sha, model, model_id, rev, weight, df)
        rows.append(r)
    if len(rows) != len(found):
        raise ValueError('Noncontiguous or duplicate validation record prefix')
    return rows


def plan_only(root: Path):
    conf, df, checkpoints, dev, view, val = inputs(root)
    path, sha = publish_validation_plan(root, val)
    # Forecast-only duration extrapolation from accepted *development* inference timings;
    # model loading and filesystem overhead NOT included. Never call this a measured ETA.
    estimates = {}
    accepted = {
        'chronos': ('c34ad91e66b4c9b0bc99500097baf6a61ea52d50bc12015ac6ca66e2239b07df',
                    '671c1151895c1ce4718ab365a16b4abcc9b1c58b347ec05f6b607c9828a528f1'),
        'moirai': ('f6dcfea50973f6f3e568ecb8320e313f5232f83806c8671818da390b40f41a27',
                   'bdd0271729fd6fb47902193836d84d3745da3e128556c15a0aef4b476943f568'),
    }
    for model in MODELS:
        # Only O(1) hash checks of already accepted Development artifacts; do
        # not repeat 1258-record audits or any GPU/data tests in TASK-006A.
        directory = dev_paths(root, DEV_PLAN_SHA, model)
        manifest_path, csv_path = directory / 'manifest.json', directory / 'predictions.csv'
        if (sha256_file(manifest_path), sha256_file(csv_path)) != accepted[model]:
            raise ValueError(f'Accepted {model} Development forecast artifact drift')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (manifest['count'] != 1258 or manifest['plan_sha256'] != DEV_PLAN_SHA or
                manifest['model'] != model or manifest['contamination_free_verified'] is not False or
                manifest['point_in_time_adjusted_prices_verified'] is not False):
            raise ValueError(f'Accepted {model} Development manifest identity drift')
        seconds = float(manifest['sum_inference_seconds_excluding_loading_replay'])
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError('Invalid original measured inference time')
        estimates[model] = {'measured_development_forecast_seconds': seconds,
            'extrapolated_validation_forecast_seconds_excluding_loading_replay':
                seconds * VAL_COUNT / 1258,
            'not_a_runtime_guarantee': True}
    return {'status': 'VALIDATION_PLAN_PUBLISHED_NO_INFERENCE',
            'plan_path': str(path), 'plan_sha256': sha,
            'forecast_count_per_model': 502, 'total_forecasts': 1004,
            'bridge_origin_index': 4530, 'first_target': val['first']['target_date'],
            'last_target': val['last']['target_date'],
            'model_estimates': estimates,
            'no_validation_forecasts_created': True,
            'contamination_free_verified': False,
            'point_in_time_adjusted_prices_verified': False, 'scientific_clearance': False}


def normalize_worker_row(row: dict) -> dict:
    """Convert NumPy scalar indices/seeds to built-in ints without mutating the locked plan.

    The plan's existing canonical representation (and SHA-256) must not change.
    Python random.seed rejects np.int64, and canonical_json_bytes(default=str)
    would otherwise serialize NumPy integer record fields as JSON strings.
    """
    return {key: int(value) if isinstance(value, np.integer) else value
            for key, value in row.items()}


def worker(root, model):
    conf, df, checkpoints, dev, view, plan = inputs(root)
    sha = digest(plan)
    plan_path = root / 'artifacts' / 'forecast_cache' / 'validation' / 'plans' / f'{sha}.json'
    if not plan_path.is_file() or plan_path.read_bytes() != canonical_json_bytes(plan):
        raise ValueError('Run requires a previously published immutable Validation plan')
    _, model_id, rev, weight, files = identity(conf, checkpoints, model)
    preflight_worker(root, conf, checkpoints, model)
    directory = val_dir(root, sha, model)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / '.validation.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Validation run locked; inspect PID before removing stale lock') from None
    os.write(fd, f'pid={os.getpid()}\n'.encode()); os.close(fd)
    try:
        existing = load_records(root, model, plan, df, model_id, rev, weight, allow_partial=True)
        adapter = None
        new = 0
        import torch
        try:
            for row in plan['origins'][len(existing):]:
                row = normalize_worker_row(row)  # runtime-only; immutable plan untouched
                if adapter is None:
                    seed_inference(2027)
                    adapter = identity(conf, checkpoints, model)[0](revision=rev, device='cuda', samples=100)
                w = validation_window(df, row['origin_index'])
                seed_inference(row['seed'])
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                start = time.perf_counter()
                q = adapter.predict(w.context)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                peak = float(torch.cuda.max_memory_allocated() / 2**20)
                diff = 0.0
                replay = row in plan['origins'][:2]
                if replay:
                    seed_inference(row['seed'])
                    q2 = adapter.predict(w.context)
                    diff = float(np.max(np.abs(q - q2)))
                    if not np.allclose(q, q2, atol=1e-5, rtol=0):
                        raise ValueError('Validation deterministic replay FAILED')
                r = {'schema_version': 'validation-forecast-v1', 'plan_sha256': sha,
                     'processed_sha256': plan['processed_sha256'],
              'processed_snapshot_id': plan['source_snapshot_id'], 'model': model,
                     'checkpoint_id': model_id, 'checkpoint_commit': rev,
                     'checkpoint_weight_sha256': weight,
                     'origin_index': row['origin_index'], 'target_index': row['target_index'],
                     'origin_date': row['origin_date'], 'target_date': row['target_date'],
                     'context_first_index': row['context_first_index'],
                     'context_last_index': row['context_last_index'],
                     'context_sha256': row['context_sha256'],
                     'target_log_return': w.target_log_return,
                     'q10': float(q[0]), 'q50': float(q[1]), 'q90': float(q[2]),
                     'seed': row['seed'], 'context_length': 256, 'horizon': 1,
                     'samples': 100, 'quantile_levels': [0.1, 0.5, 0.9],
                     'target_units': 'daily_log_return_decimal',
                     'elapsed_seconds': elapsed, 'peak_allocated_mib': peak,
                     'replay_checked': replay, 'replay_max_abs_diff': diff,
                     'contamination_free_verified': False,
                     'point_in_time_adjusted_prices_verified': False}
                record_validate(r, row, plan, sha, model, model_id, rev, weight, df)
                write_once_atomic(directory / 'records' / f'{row["origin_index"]:06d}.json',
                                  canonical_json_bytes(r))
                new += 1
                if new % 50 == 0:
                    print(json.dumps({'progress': model, 'created': new,
                                      'latest_origin': row['origin_index']}), flush=True)
        finally:
            if adapter is not None:
                del adapter
                torch.cuda.empty_cache()
        complete = load_records(root, model, plan, df, model_id, rev, weight, allow_partial=False)
        csv_bytes = pd.DataFrame.from_records(complete).to_csv(index=False, float_format='%.17g').encode('utf-8')
        csv_path = directory / 'predictions.csv'
        if csv_path.exists():
            if csv_path.read_bytes() != csv_bytes:
                raise ValueError('Validation CSV differs from records')
        else:
            write_once_atomic(csv_path, csv_bytes)
        manifest = {'status': 'VALIDATION_FORECAST_CACHE_COMPLETE_NOT_SCIENTIFIC_CLEARANCE',
            'plan_sha256': sha, 'model': model, 'count': VAL_COUNT,
            'checkpoint_id': model_id, 'checkpoint_commit': rev,
            'checkpoint_weight_sha256': weight,
            'predictions_sha256': sha256_file(csv_path),
            'record_sha256': {f'{r["origin_index"]:06d}.json': sha256_file(directory / 'records' / f'{r["origin_index"]:06d}.json') for r in complete},
            'contamination_free_verified': False,
            'point_in_time_adjusted_prices_verified': False,
            'scientific_clearance': False}
        store_or_verify(directory / 'manifest.json', manifest)
        return {'status': 'VALIDATION_MODEL_CACHE_COMPLETE', 'model': model,
                'new_records': new, 'count': len(complete), 'plan_sha256': sha}
    finally:
        lock.unlink(missing_ok=True)


def audit_model(root, model, conf, df, checkpoints, plan):
    _, model_id, rev, weight, _ = identity(conf, checkpoints, model)
    directory = val_dir(root, digest(plan), model)
    rows = load_records(root, model, plan, df, model_id, rev, weight, allow_partial=False)
    csv_bytes = pd.DataFrame.from_records(rows).to_csv(index=False, float_format='%.17g').encode('utf-8')
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if not (directory / 'manifest.json').read_bytes() == canonical_json_bytes(manifest):
        raise ValueError('Validation manifest not canonical')
    if (manifest['plan_sha256'] != digest(plan) or manifest['count'] != VAL_COUNT or
        manifest['model'] != model or manifest['checkpoint_id'] != model_id or
        manifest['status'] != 'VALIDATION_FORECAST_CACHE_COMPLETE_NOT_SCIENTIFIC_CLEARANCE' or
        any(manifest.get(k) is not False for k in ('contamination_free_verified',
                'point_in_time_adjusted_prices_verified', 'scientific_clearance')) or
        manifest['checkpoint_commit'] != rev or manifest['checkpoint_weight_sha256'] != weight or
        manifest['predictions_sha256'] != sha256_file(directory / 'predictions.csv') or
        (directory / 'predictions.csv').read_bytes() != csv_bytes or
        manifest['record_sha256'] != {f'{r["origin_index"]:06d}.json': sha256_file(directory / 'records' / f'{r["origin_index"]:06d}.json') for r in rows}):
        raise ValueError('Validation manifest/records mismatch')
    return rows, {'manifest_sha256': sha256_file(directory / 'manifest.json'),
                  'predictions_sha256': manifest['predictions_sha256']}


def evaluate(root: Path):
    conf, df, checkpoints, dev, view, plan = inputs(root)
    sha = digest(plan)
    published = root / 'artifacts' / 'forecast_cache' / 'validation' / 'plans' / f'{sha}.json'
    if not published.is_file() or published.read_bytes() != canonical_json_bytes(plan):
        raise ValueError('Validation plan absent or changed after publication')
    vrows, drows, prov = {}, {}, {}
    for model in MODELS:
        _, mid, rev, weight, _ = identity(conf, checkpoints, model)
        _, drows[model] = verify_complete(dev_paths(root, DEV_PLAN_SHA, model), view, DEV_PLAN_SHA, model, mid, rev, weight, df)
        vrows[model], prov[model] = audit_model(root, model, conf, df, checkpoints, plan)
        check_boundary_records(drows[model], vrows[model], dev, plan)
    validate_cross_model(vrows['chronos'], vrows['moirai'])
    # Entire 1258+502 stream replays causal online states; no reset at boundary.
    initial = make_window(df, dev['origins'][0]['origin_index']).context
    p = Parameters()
    if (p.alpha, p.window, p.gamma, p.decay, p.correction) != (0.2, 252, 0.01, 0.94, 'expansion_only'):
        raise ValueError('Calibration constants drift')
    results = {}
    rows_out = []
    # An accepted development trace is the byte-verified anchor for carry-forward.
    dev_report_path = root / 'artifacts' / 'calibration' / 'task005b' / DEV_PLAN_SHA / 'full_development_W252_expansion_only.json'
    if sha256_file(dev_report_path) != ACCEPTED_DEV_REPORT_SHA:
        raise ValueError('Accepted Development statistical report SHA drift')
    dev_report = json.loads(dev_report_path.read_text(encoding='utf-8'))
    if dev_report['diagnostic_trace_sha256'] != ACCEPTED_DEV_TRACE_SHA:
        raise ValueError('Accepted Development statistical trace identity drift')
    trace_path = root / dev_report['diagnostic_trace_path']
    if (dev_report['full_plan_sha256'] != DEV_PLAN_SHA or
            sha256_file(trace_path) != dev_report['diagnostic_trace_sha256']):
        raise ValueError('Previously accepted development calibration trace has drifted')
    development_trace = pd.read_csv(trace_path)
    mask = None
    for model in MODELS:
        records = drows[model] + vrows[model]
        regime = causal_regime_indices(records, initial)
        if len(regime['high']) != 1508:  # 1258+502-252
            raise ValueError('Regime history cardinality drift')
        vmask = regime['high'][-VAL_COUNT:]
        if mask is not None and not np.array_equal(mask, vmask):
            raise ValueError('Cross-backbone regime mask mismatch')
        mask = vmask
        methods = {}
        for scale in SCALES:
            for controller in CONTROLLERS:
                cell = f'{scale}__{controller}'
                full = calibrate(records, initial, p, scale, controller)
                if len(full) != 1508:
                    raise ValueError('Calibration stream unexpectedly reset or truncated')
                previous = development_trace[(development_trace['model'] == model) &
                                             (development_trace['cell'] == cell)].sort_values('origin_index')
                if len(previous) != 1006:
                    raise ValueError('Missing accepted Development calibration prefix')
                for key in ('lower', 'upper', 'scale', 'alpha_t', 'gamma_t'):
                    observed = np.asarray([row[key] for row in full[:1006]], dtype=float)
                    if not np.allclose(observed, previous[key].to_numpy(dtype=float), atol=1e-12, rtol=0):
                        raise ValueError(f'Carried calibration prefix disagrees with locked Development trace: {cell}/{key}')
                val_rows = full[-VAL_COUNT:]
                if [x['origin_index'] for x in val_rows] != [x['origin_index'] for x in vrows[model]]:
                    raise ValueError('Validation origin mismatch')
                metrics = interval_metrics(val_rows, p.alpha)
                arrays = score_arrays(val_rows, p.alpha)
                regime_metrics = {}
                for key, selection in [('high', vmask), ('other', ~vmask)]:
                    ix = np.flatnonzero(selection)
                    regime_metrics[key] = {'n': len(ix), **{k: float(np.mean(a[ix])) if len(ix) else None for k, a in arrays.items()}}
                methods[cell] = {'metrics': metrics, 'regimes': regime_metrics}
                for j, r in enumerate(val_rows):
                    rows_out.append({'model': model, 'cell': cell, **r,
                                     'regime': 'high' if vmask[j] else 'other'})
        # Native q10-q90, reported but never fitted.
        raw = [{'lower': r['q10'], 'upper': r['q90'], 'target_log_return': r['target_log_return']} for r in vrows[model]]
        methods['raw_q10_q90'] = {'metrics': interval_metrics(raw, p.alpha)}
        all_scores = [signed_score(r['target_log_return'], r['q10'], r['q90']) for r in records]
        signed_rows, bad = [], []
        for j in range(len(drows[model]), len(records)):
            rec = records[j]
            q = conformal_order_statistic(all_scores[j - p.window:j], p.alpha)
            lo, hi = rec['q10'] - q, rec['q90'] + q
            if not np.isfinite([lo, hi]).all() or lo > hi:
                bad.append(rec['origin_index'])
            else:
                signed_rows.append({'lower': lo, 'upper': hi,
                                    'target_log_return': rec['target_log_return']})
        methods['signed_rolling_none'] = {
            'status': 'VALIDATION_EMPTY_SIGNED_SET' if bad else 'FINITE_SIGNED_INTERVALS',
            'empty_origins_in_validation': bad,
            'coverage_with_empty_as_miss': (sum(r['lower'] <= r['target_log_return'] <= r['upper'] for r in signed_rows) / VAL_COUNT) if bad else None,
            'metrics': interval_metrics(signed_rows, p.alpha) if not bad else None,
            'signed_winkler_undefined_if_empty': bool(bad)}
        # Regime-specific uncertainty uses the same chronological bootstrap indices,
        # with regime labels resampled together with outcomes; no within-regime shuffling.
        regime_draws = circular_block_indices(VAL_COUNT, 1000, 20, 2027)
        for cell in (f'{s}__{c}' for s in SCALES for c in CONTROLLERS):
            cell_rows = [r for r in rows_out if r['model'] == model and r['cell'] == cell]
            arrays = score_arrays(cell_rows, p.alpha)
            for name, is_high in [('high', True), ('other', False)]:
                selection = vmask if is_high else ~vmask
                selected = selection[regime_draws]
                counts = selected.sum(axis=1)
                valid = counts > 0
                details = methods[cell]['regimes'][name]
                details['ci_method'] = 'paired_circular_blocks_20_unadjusted_exploratory'
                details['valid_bootstrap_replicates'] = int(valid.sum())
                details['empty_bootstrap_replicates'] = int((~valid).sum())
                details['ci95'] = {}
                for metric, values in arrays.items():
                    means = (values[regime_draws][valid] * selected[valid]).sum(axis=1) / counts[valid]
                    details['ci95'][metric] = ([float(v) for v in np.quantile(means, [0.025, 0.975])]
                                               if len(means) else None)
        results[model] = methods
    # Compute paired moving-block CI against preregistered unscaled rolling comparator.
    out_frame = pd.DataFrame.from_records(rows_out)
    baseline = 'none__rolling_cqr'
    for L in (10, 20, 40):
        draws = circular_block_indices(VAL_COUNT, 1000, L, 2027)
        for model in MODELS:
            by_cell = {c: g.sort_values('origin_index') for c, g in out_frame[out_frame.model == model].groupby('cell')}
            base = score_arrays(by_cell[baseline].to_dict('records'), p.alpha)['winkler']
            for scale in SCALES:
                fixed = score_arrays(by_cell[f'{scale}__fixed_gamma_aci'].to_dict('records'), p.alpha)['winkler']
                for controller in CONTROLLERS:
                    cell = f'{scale}__{controller}'
                    cur = score_arrays(by_cell[cell].to_dict('records'), p.alpha)['winkler']
                    for name, comparator in [('unscaled_rolling', base), ('same_scale_fixed', fixed)]:
                        diff = cur - comparator
                        qs = np.quantile(diff[draws].mean(axis=1), [0.025, 0.975])
                        results[model][cell].setdefault('paired_contrasts', {}).setdefault(name, {})[str(L)] = {
                            'mean': float(diff.mean()), 'ci95': [float(qs[0]), float(qs[1])],
                            'exploratory': True, 'multiplicity_adjusted': False}
    out_dir = root / 'artifacts' / 'calibration' / 'validation' / sha
    csv_bytes = out_frame.to_csv(index=False, float_format='%.17g').encode('utf-8')
    trace_path = out_dir / 'validation_502_per_cell.csv'
    if trace_path.exists():
        if trace_path.read_bytes() != csv_bytes:
            raise ValueError('Validation trace bytes differ on rerun')
    else:
        write_once_atomic(trace_path, csv_bytes)
    report = {'status': 'VALIDATION_CALIBRATION_AUDIT_PASS_NOT_SCIENTIFIC_CLEARANCE',
              'validation_plan_sha256': sha, 'development_plan_sha256': DEV_PLAN_SHA,
              'records_per_model': VAL_COUNT, 'calibration_state': plan['calibration_state'],
              'trace_sha256': sha256_file(trace_path), 'forecast_inputs': prov,
              'regime_counts': {'high': int(mask.sum()), 'other': int((~mask).sum())},
              'models': results,
              'method_selection_performed': False, 'validation_hyperparameter_tuning': False,
              'final_test_used': False, 'contamination_free_verified': False,
              'point_in_time_adjusted_prices_verified': False, 'scientific_clearance': False}
    report_path = out_dir / 'validation_calibration_report.json'
    store_or_verify(report_path, report)
    return {'status': report['status'], 'report_path': str(report_path),
            'report_sha256': sha256_file(report_path), 'plan_sha256': sha,
            'evaluation_n_per_cell': VAL_COUNT, 'regime_counts': report['regime_counts'],
            'scientific_clearance': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run', '_worker', 'audit'])
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--allow-validation', action='store_true')
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'plan':
        result = plan_only(root)
    elif args.command == '_worker':
        if not args.allow_validation or not args.model:
            raise ValueError('Direct worker requires explicit model + authorization')
        result = worker(root, args.model)
    elif args.command == 'audit':
        result = evaluate(root)
    else:
        if not args.allow_validation:
            raise ValueError('Validation GPU execution requires --allow-validation')
        result = plan_only(root)  # plan and dev audit only, not GPU
        # CPU tests first, no repeated accepted GPU tests.
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_task006a_boundary.py'],
                       cwd=root, check=True)
        for model in MODELS:
            interp = root / ('.venv' if model == 'chronos' else '.venv-moirai') / 'Scripts' / 'python.exe'
            if not interp.exists():
                raise FileNotFoundError(f'Missing validated model-specific interpreter: {interp}')
            command = [str(interp), str(Path(__file__).resolve()), '_worker', '--root', str(root),
                       '--model', model, '--allow-validation']
            env = {**os.environ, 'PYTHONPATH': str(root / 'src')}
            subprocess.run(command, cwd=root, env=env, check=True)
        result = evaluate(root)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
