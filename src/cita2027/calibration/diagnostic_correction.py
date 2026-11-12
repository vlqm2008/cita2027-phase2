"""TASK-005B.1: read-only, post-hoc diagnostics of accepted development forecasts.

Never changes TASK-005B output or core factorial protocol. Returns reported here
are explanatory and do not qualify as independent validation/test evidence.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cita2027.calibration.online import Parameters, SCALES, CONTROLLERS, calibrate
from cita2027.calibration.development_analysis import (
    circular_block_indices, paired_ci, score_arrays, distribution,
)
from cita2027.utils.provenance import sha256_file

PLAN_SHA = "39c14b72d7bd82b815757147838e45fe3df56bddf4616367139cfd5661bd089f"
REPORT_SHA = "972a196f7beae4e31a343a1163beb678f76ee29e63362bb9b10171d3ea1ff149"
REGIME_QUANTILE = 0.75
REGIME_HISTORY = 252
BLOCK_LENGTHS = (10, 20, 40)
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 2027
N_EVAL = 1006


def causal_regime_indices(records: list[dict], initial_context: np.ndarray,
                          window: int = REGIME_HISTORY,
                          quantile: float = REGIME_QUANTILE,
                          decay: float = 0.94,
                          min_scale: float = 1e-6) -> dict:
    """Classify origin i using volatility at i and quantile of volatilities at origins <i.

    Initialization is RMS of exactly 256 returns through FIRST origin, then
    updates occur only AFTER the target from the current origin is observed.
    Prior sigma history must have at least window entries; no substitution
    from evaluation sample or realized future returns is allowed.
    """
    x = np.asarray(initial_context, dtype=float)
    if (x.shape != (256,) or not np.isfinite(x).all() or len(records) <= window
            or window < 1 or not 0 < quantile < 1 or not 0 < decay < 1):
        raise ValueError("Invalid causal regime inputs")
    var = max(float(np.mean(x**2)), min_scale**2)
    prior = []
    eval_sigmas, thresholds, high = [], [], []
    for j, r in enumerate(records):
        if j and r['origin_index'] != records[j - 1]['origin_index'] + 1:
            raise ValueError('Noncontiguous records')
        if (r['target_index'] != r['origin_index'] + 1
                or r['seed'] != 2027 + r['origin_index']):
            raise ValueError('Index/seed drift')
        sigma = max(float(np.sqrt(var)), min_scale)
        if j >= window:
            if len(prior) < window:
                raise ValueError('Insufficient causal volatility history')
            q = float(np.quantile(prior[-window:], quantile, method='linear'))
            thresholds.append(q)
            eval_sigmas.append(sigma)
            high.append(bool(sigma > q))  # equality classified as other
        prior.append(sigma)
        y = float(r['target_log_return'])
        if not np.isfinite(y):
            raise ValueError('Nonfinite observed return')
        var = decay * var + (1 - decay) * y * y
    return {'high': np.asarray(high, dtype=bool),
            'sigma': np.asarray(eval_sigmas, dtype=float),
            'threshold': np.asarray(thresholds, dtype=float)}


def load_traces(trace_path: Path, report: dict) -> dict[str, dict[str, pd.DataFrame]]:
    if sha256_file(trace_path) != report['diagnostic_trace_sha256']:
        raise ValueError('Accepted per-origin trace SHA mismatch')
    trace = pd.read_csv(trace_path)
    needed = {'model', 'cell', 'origin_index', 'target_index', 'origin_date',
              'target_date', 'target_log_return', 'lower', 'upper', 'winkler',
              'scale', 'alpha_t', 'gamma_t', 'gamma_driver', 'return_volatility'}
    if not needed <= set(trace.columns):
        raise ValueError(f'Missing trace columns: {needed - set(trace.columns)}')
    if len(trace) != 2 * len(SCALES) * len(CONTROLLERS) * N_EVAL:
        raise ValueError('Wrong trace row count')
    out = {}
    expected_models = {'chronos', 'moirai'}
    if set(trace.model) != expected_models:
        raise ValueError('Unexpected trace model')
    for model, frame in trace.groupby('model', sort=False):
        cells = {}
        expected = {f'{s}__{c}' for s in SCALES for c in CONTROLLERS}
        if set(frame.cell) != expected:
            raise ValueError('Missing/extra factorial cells')
        for cell, d in frame.groupby('cell', sort=False):
            d = d.sort_values('origin_index', kind='stable').reset_index(drop=True)
            if len(d) != N_EVAL or d.origin_index.duplicated().any():
                raise ValueError('Missing/duplicate origins')
            if not np.array_equal(d.origin_index.to_numpy(), np.arange(3524, 4530)):
                raise ValueError('Origin range drift')
            if not np.array_equal(d.target_index.to_numpy(), d.origin_index.to_numpy() + 1):
                raise ValueError('Target index drift')
            if not np.isfinite(d[['lower', 'upper', 'target_log_return', 'winkler',
                                  'gamma_driver', 'return_volatility']].to_numpy(float)).all():
                raise ValueError('Nonfinite traces')
            arr = score_arrays(d.to_dict('records'), .2)
            if not np.allclose(arr['winkler'], d.winkler.to_numpy(float), rtol=1e-10, atol=1e-12):
                raise ValueError('Trace Winkler mismatch')
            documented = report['models'][model]['cells'][cell]['metrics']['winkler_mean']
            if abs(float(arr['winkler'].mean()) - documented) > 1e-10:
                raise ValueError('Accepted report mean mismatch')
            cells[cell] = d
        key = cells['none__rolling_cqr']
        for d in cells.values():
            for col in ('origin_index', 'target_index', 'origin_date', 'target_date', 'target_log_return'):
                if not np.array_equal(d[col].to_numpy(), key[col].to_numpy()):
                    raise ValueError(f'Within-model trace mismatch: {col}')
        out[model] = cells
    for model, cells in out.items():
        shared = cells['none__variable_gamma_aci'].gamma_driver.to_numpy(float)
        for d in cells.values():
            if not np.array_equal(d.gamma_driver.to_numpy(float), shared):
                raise ValueError(f'Gamma driver differs between factorial conditions: {model}')
    if not np.array_equal(out['chronos']['none__variable_gamma_aci'].gamma_driver.to_numpy(float),
                          out['moirai']['none__variable_gamma_aci'].gamma_driver.to_numpy(float)):
        raise ValueError('Gamma driver differs between models')
    for col in ('origin_index', 'target_index', 'origin_date', 'target_date', 'target_log_return'):
        if not np.array_equal(out['chronos']['none__rolling_cqr'][col].to_numpy(),
                              out['moirai']['none__rolling_cqr'][col].to_numpy()):
            raise ValueError('Cross-model evaluation trace mismatch')
    return out


def masked_bootstrap_ci(values: np.ndarray, mask: np.ndarray, indices: np.ndarray) -> dict:
    """Circular block bootstrap of a fixed, previsible regime indicator and outcome.

    Each resampled observation keeps its regime membership; avoid using a
    retrospectively optimized threshold. Replicates with empty groups are
    excluded explicitly and counted, never filled with zero.
    """
    v = np.asarray(values, float)
    mask = np.asarray(mask, bool)
    if (v.shape != mask.shape or v.ndim != 1 or indices.shape[1] != len(v)
            or not mask.any() or not np.isfinite(v).all()):
        return {'n': int(mask.sum()), 'mean': None, 'ci_95': None,
                'valid_replicates': 0, 'empty_replicates': len(indices)}
    picked = mask[indices]
    counts = picked.sum(axis=1)
    ok = counts > 0
    means = (v[indices] * picked).sum(axis=1)[ok] / counts[ok]
    ci = np.quantile(means, [.025, .975], method='linear') if len(means) else [np.nan, np.nan]
    return {'n': int(mask.sum()), 'mean': float(v[mask].mean()),
            'ci_95': list(map(float, ci)) if len(means) else None,
            'valid_replicates': int(ok.sum()), 'empty_replicates': int((~ok).sum())}


def gamma_diagnostics(d: pd.DataFrame, params: Parameters) -> dict:
    driver = d.gamma_driver.to_numpy(float)
    gamma = d.gamma_t.to_numpy(float)
    if np.any(~np.isfinite(driver)) or np.any(~np.isfinite(gamma)):
        raise ValueError('Nonfinite gamma driver')
    if np.any((driver < params.driver_min - 1e-12) | (driver > params.driver_max + 1e-12)):
        raise ValueError('Gamma driver clipping bounds violation')
    expected = (params.gamma * driver if d.cell.iat[0].endswith('variable_gamma_aci')
                else np.zeros_like(driver) if d.cell.iat[0].endswith('rolling_cqr')
                else np.full_like(driver, params.gamma))
    if not np.allclose(gamma, expected, atol=1e-12, rtol=1e-12):
        raise ValueError('Effective gamma mismatch')
    years = pd.to_datetime(d.origin_date).dt.year.to_numpy()
    upper = np.isclose(driver, params.driver_max, atol=1e-12, rtol=0)
    longest = run = episodes = 0
    for flag in upper:
        if flag:
            if run == 0:
                episodes += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return {'driver_distribution': distribution(driver.tolist()),
            'driver_at_min_fraction': float(np.mean(np.isclose(driver, params.driver_min, atol=1e-12, rtol=0))),
            'driver_at_max_fraction': float(np.mean(upper)),
            'upper_saturation_episodes': episodes, 'longest_upper_saturation_run_origins': longest,
            'effective_gamma_distribution': distribution(gamma.tolist()),
            'annual': {str(int(year)): {'n': int(np.sum(years == year)),
                                       'at_upper_fraction': float(np.mean(np.isclose(driver[years == year], params.driver_max, atol=1e-12, rtol=0))),
                                       'mean_gamma': float(np.mean(gamma[years == year]))}
                       for year in np.unique(years)}}


def evaluate_report(accepted: dict, traces: dict[str, dict[str, pd.DataFrame]],
                    records: dict[str, list[dict]], initial_context: np.ndarray,
                    params: Parameters | None = None, bootstrap_reps: int = BOOTSTRAP_REPS) -> dict:
    """All primary cells read from immutable traces; only fixed-gamma=.02 recalibrated."""
    params = params or Parameters()
    if (params.window != 252 or params.alpha != .2 or params.gamma != .01
            or params.correction != 'expansion_only' or bootstrap_reps < 1):
        raise ValueError('TASK-005B primary protocol drift')
    regime = causal_regime_indices(records['chronos'], initial_context,
                                   window=params.window, decay=params.decay,
                                   min_scale=params.min_scale)
    if not np.array_equal([r['target_log_return'] for r in records['chronos']],
                          [r['target_log_return'] for r in records['moirai']]):
        raise ValueError('Cross-model realized targets differ')
    for model in ('chronos', 'moirai'):
        if len(records[model]) != 1258:
            raise ValueError('Cache completeness drift')
        if not np.allclose(traces[model]['none__rolling_cqr'].return_volatility.to_numpy(float),
                               regime['sigma'], rtol=1e-12, atol=1e-14):
            raise ValueError('Trace volatility differs from causal recursion')
    masks = {'high': regime['high'], 'other': ~regime['high']}
    result = {
        'task': 'TASK-005B.1', 'status': 'DIAGNOSTICS_PASS_NOT_SCIENTIFIC_CLEARANCE',
        'base_report_sha256': REPORT_SHA, 'full_plan_sha256': PLAN_SHA,
        'method': {'regime_threshold': 'rolling_prior_252_origin_return_EWMA_sigma_quantile_0.75',
                   'regime_quantile': REGIME_QUANTILE, 'min_history': REGIME_HISTORY,
                   'tie_rule': 'sigma_t > threshold_t is high; equality is other',
                   'bootstrap': 'paired_circular_moving_blocks', 'block_lengths': list(BLOCK_LENGTHS),
                   'replicates': bootstrap_reps, 'seed': BOOTSTRAP_SEED,
                   'multiple_comparison_adjustment': None, 'inference': 'exploratory',
                   'fixed_gamma_0_02': 'post_hoc_explanatory_not_a_main_factorial_cell'},
        'n_eval': N_EVAL,
        'regime_counts': {name: int(mask.sum()) for name, mask in masks.items()},
        'regime_threshold_distribution': distribution(regime['threshold'].tolist()),
        'models': {}, 'validation_used': False, 'final_test_used': False,
        'contamination_free_verified': False, 'point_in_time_adjusted_prices_verified': False,
        'scientific_clearance': False,
    }
    if min(result['regime_counts'].values()) == 0:
        raise ValueError('Regime segmentation still has an empty group')
    draws = {length: circular_block_indices(N_EVAL, bootstrap_reps, length, BOOTSTRAP_SEED)
             for length in BLOCK_LENGTHS}
    fields = ('winkler', 'covered', 'width', 'lower_miss', 'upper_miss', 'lower_penalty', 'upper_penalty')
    for model in ('chronos', 'moirai'):
        md = {'regimes': {}, 'regime_paired_vs_none_rolling': {}, 'gamma': {}, 'direct_paired_contrasts': {},
              'fixed_gamma_0_02_sensitivity': {}}
        scores = {}
        for scale in SCALES:
            for ctrl in CONTROLLERS:
                key = f'{scale}__{ctrl}'
                d = traces[model][key]
                arrays = score_arrays(d.to_dict('records'), params.alpha)
                scores[key] = arrays
                md['gamma'][key] = gamma_diagnostics(d, params)
                md['regimes'][key] = {
                    group: {field: masked_bootstrap_ci(arrays[field], mask, draws[20])
                            for field in fields}
                    for group, mask in masks.items()}
        for key, arrays in scores.items():
            if key == 'none__rolling_cqr':
                continue
            diff = arrays['winkler'] - scores['none__rolling_cqr']['winkler']
            md['regime_paired_vs_none_rolling'][key] = {
                group: masked_bootstrap_ci(diff, mask, draws[20])
                for group, mask in masks.items()}
        for scale in SCALES:
            for left, right in (
                (f'{scale}__variable_gamma_aci', f'{scale}__fixed_gamma_aci'),
                (f'{scale}__variable_gamma_aci', f'{scale}__rolling_cqr'),
                (f'{scale}__fixed_gamma_aci', f'{scale}__rolling_cqr'),
            ):
                dif = scores[left]['winkler'] - scores[right]['winkler']
                md['direct_paired_contrasts'][f'{left}_minus_{right}'] = {
                    str(l): paired_ci(dif, indices) for l, indices in draws.items()}
        for scale in SCALES[1:]:
            for ctrl in CONTROLLERS:
                left = f'{scale}__{ctrl}'
                right = f'none__{ctrl}'
                dif = scores[left]['winkler'] - scores[right]['winkler']
                md['direct_paired_contrasts'][f'{left}_minus_{right}'] = {
                    str(l): paired_ci(dif, indices) for l, indices in draws.items()}
        for scale in SCALES:
            rows = calibrate(records[model], initial_context, replace(params, gamma=0.02),
                             scale, 'fixed_gamma_aci')
            d = score_arrays(rows, params.alpha)
            if not np.array_equal([r['origin_index'] for r in rows],
                                  traces[model][f'{scale}__fixed_gamma_aci'].origin_index.to_numpy()):
                raise ValueError('Fixed gamma sensitivity origin drift')
            for right in (f'{scale}__variable_gamma_aci', f'{scale}__fixed_gamma_aci'):
                delta = d['winkler'] - scores[right]['winkler']
                md['fixed_gamma_0_02_sensitivity'][f'{scale}__fixed_0_02_minus_{right}'] = {
                    str(l): paired_ci(delta, indices) for l, indices in draws.items()}
            md['fixed_gamma_0_02_sensitivity'][f'{scale}__metrics'] = {
                'winkler_mean': float(d['winkler'].mean()),
                'coverage': float(d['covered'].mean()),
                'width_mean': float(d['width'].mean()),
                'lower_miss_rate': float(d['lower_miss'].mean()),
                'upper_miss_rate': float(d['upper_miss'].mean())}
        result['models'][model] = md
    return result


def validate_input_report(path: Path) -> dict:
    if sha256_file(path) != REPORT_SHA:
        raise ValueError('Accepted TASK-005B JSON SHA mismatch')
    report = json.loads(path.read_bytes())
    if (report['full_plan_sha256'] != PLAN_SHA or report['protocol']['evaluation_origins'] != N_EVAL
            or report['protocol']['window'] != 252 or report['protocol']['gamma'] != .01
            or report['protocol']['nominal_miscoverage'] != .2
            or report['status'] != 'FULL_DEVELOPMENT_CALIBRATION_FILE_AUDIT_PASS_NOT_SCIENTIFIC_CLEARANCE'
            or report['contamination_free_verified'] is not False
            or report['point_in_time_adjusted_prices_verified'] is not False
            or report['scientific_clearance'] is not False):
        raise ValueError('Accepted TASK-005B report/protocol drift')
    return report
