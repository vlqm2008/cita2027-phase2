"""TASK-005B: read-only comparative diagnostics for audited development forecasts.

Inference is never invoked here. All bootstraps resample paired *time indices*.
All statistics are exploratory and require caution under serial dependence.
"""
from __future__ import annotations

from dataclasses import asdict
from math import ceil

import numpy as np

from cita2027.calibration.online import (
    CONTROLLERS, SCALES, Parameters, calibrate, conformal_order_statistic,
    interval_metrics, signed_score,
)

BLOCK_LENGTH = 20  # Prespecified primary setting (20 exchange sessions).
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 2027


def validated_pair_keys(rows: list[dict]) -> list[tuple[int, int, str, str]]:
    keys = [(r['origin_index'], r['target_index'], r['origin_date'], r['target_date']) for r in rows]
    if len(set(keys)) != len(keys) or not all(b == a + 1 for a, b, _, _ in keys):
        raise ValueError('Duplicate/misaligned origin-target keys')
    if not all(keys[i + 1][0] == keys[i][0] + 1 for i in range(len(keys) - 1)):
        raise ValueError('Noncontiguous evaluation origins')
    return keys


def score_arrays(rows: list[dict], alpha: float) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError('Empty evaluation')
    lo = np.asarray([r['lower'] for r in rows], dtype=np.float64)
    hi = np.asarray([r['upper'] for r in rows], dtype=np.float64)
    y = np.asarray([r['target_log_return'] for r in rows], dtype=np.float64)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all() and np.isfinite(y).all() and (lo <= hi).all()):
        raise ValueError('Nonfinite or inverted interval')
    lower_miss = y < lo
    upper_miss = y > hi
    width = hi - lo
    lower_penalty = (2 / alpha) * (lo - y) * lower_miss
    upper_penalty = (2 / alpha) * (y - hi) * upper_miss
    return {'winkler': width + lower_penalty + upper_penalty,
            'width': width, 'lower_penalty': lower_penalty, 'upper_penalty': upper_penalty,
            'covered': (~(lower_miss | upper_miss)).astype(float),
            'lower_miss': lower_miss.astype(float), 'upper_miss': upper_miss.astype(float)}


def native_rows(records: list[dict], window: int) -> list[dict]:
    return [{'origin_index': r['origin_index'], 'target_index': r['target_index'],
             'origin_date': r['origin_date'], 'target_date': r['target_date'],
             'target_log_return': r['target_log_return'], 'lower': r['q10'], 'upper': r['q90']}
            for r in records[window:]]


def signed_rolling_diagnostic(records: list[dict], window: int, alpha: float) -> dict:
    """Unmodified signed-score CQR threshold (no clamp), including empty-set diagnostics.

    A negative threshold exceeding half the native interval width gives an empty set.
    Its Winkler score is undefined; never silently discard such origins.
    """
    if len(records) <= window:
        raise ValueError('Insufficient records')
    scores = [signed_score(r['target_log_return'], r['q10'], r['q90']) for r in records]
    rows, empty = [], []
    for j in range(window, len(records)):
        r = records[j]
        q = conformal_order_statistic(scores[j - window:j], alpha)
        lo, up = r['q10'] - q, r['q90'] + q
        if lo > up or not np.isfinite([lo, up]).all():
            empty.append({'origin_index': r['origin_index'], 'threshold': q,
                          'raw_width': r['q90'] - r['q10']})
        else:
            rows.append({'origin_index': r['origin_index'], 'target_index': r['target_index'],
                         'origin_date': r['origin_date'], 'target_date': r['target_date'],
                         'target_log_return': r['target_log_return'], 'lower': lo, 'upper': up})
    if empty:
        # Correct full-sample empty-set coverage; ordinary Winkler is NOT defined.
        covered = sum(r['lower'] <= r['target_log_return'] <= r['upper'] for r in rows)
        return {'status': 'EMPTY_CONFORMAL_SET_WINKLER_UNDEFINED',
                'evaluated': len(records) - window, 'empty_count': len(empty),
                'empty_origin_indices': [item['origin_index'] for item in empty],
                'coverage_with_empty_as_miss': covered / (len(records) - window),
                'metrics': None, 'rows': None}
    return {'status': 'FINITE_SIGNED_INTERVALS', 'evaluated': len(rows), 'empty_count': 0,
            'empty_origin_indices': [], 'coverage_with_empty_as_miss': None,
            'metrics': interval_metrics(rows, alpha), 'rows': rows}


def circular_block_indices(n: int, reps: int = BOOTSTRAP_REPS, length: int = BLOCK_LENGTH,
                           seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Circular moving-block bootstrap with fixed-length contiguous trading-session blocks."""
    if n <= 0 or reps < 1 or length < 1 or length > n:
        raise ValueError('Invalid bootstrap size')
    rng = np.random.default_rng(seed)
    blocks = ceil(n / length)
    starts = rng.integers(0, n, size=(reps, blocks))
    return ((starts[:, :, None] + np.arange(length)) % n).reshape(reps, -1)[:, :n]


def paired_ci(difference: np.ndarray, indices: np.ndarray) -> dict:
    d = np.asarray(difference, dtype=np.float64)
    if d.ndim != 1 or len(d) != indices.shape[1] or not np.isfinite(d).all():
        raise ValueError('Misaligned/nonfinite paired differences')
    rep_means = np.mean(d[indices], axis=1)
    left, right = np.quantile(rep_means, [0.025, 0.975], method='linear')
    return {'mean_difference': float(np.mean(d)), 'ci_95_percentile': [float(left), float(right)],
            'ci_excludes_zero': bool(left > 0 or right < 0), 'n': len(d),
            'lower_is_better': True}


def distribution(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    if not a.size or not np.isfinite(a).all():
        raise ValueError('Empty/nonfinite diagnostic distribution')
    return {'min': float(a.min()), 'p05': float(np.quantile(a, .05)),
            'median': float(np.median(a)), 'p95': float(np.quantile(a, .95)),
            'max': float(a.max())}


def diagnostics(rows: list[dict], params: Parameters, reference: float) -> dict:
    floor = 1 / (params.window + 1)
    at_low = sum(r['alpha_t'] <= floor + 1e-12 for r in rows)
    at_high = sum(r['alpha_t'] >= 1 - floor - 1e-12 for r in rows)
    would_clip_low = would_clip_high = 0
    for r in rows:
        if r['gamma_t']:
            raw_next = r['alpha_t'] + r['gamma_t'] * (params.alpha - r['miss'])
            would_clip_low += raw_next < floor
            would_clip_high += raw_next > 1 - floor
    high = [r for r in rows if r['return_volatility'] > reference]
    other = [r for r in rows if r['return_volatility'] <= reference]
    return {
        'alpha_at_lower_bound': at_low, 'alpha_at_upper_bound': at_high,
        'alpha_next_clip_lower_events': would_clip_low,
        'alpha_next_clip_upper_events': would_clip_high,
        'scale': distribution([r['scale'] for r in rows]),
        'gamma_driver': distribution([r['gamma_driver'] for r in rows]),
        'return_volatility': distribution([r['return_volatility'] for r in rows]),
        'volatility_regimes': {
            'threshold': reference, 'threshold_source': 'first_origin_256_return_rms',
            'high': {'n': len(high), 'metrics': interval_metrics(high, params.alpha) if high else None},
            'other': {'n': len(other), 'metrics': interval_metrics(other, params.alpha) if other else None},
            'conditional_coverage_guarantee': False,
        },
    }


def analyze_model(records: list[dict], initial_context: np.ndarray, params: Parameters,
                  bootstrap_indices: np.ndarray) -> tuple[dict, list[dict]]:
    if len(records) != 1258 or params.window != 252 or params.correction != 'expansion_only':
        raise ValueError('Accepted primary analysis protocol drift')
    evaluation_n = len(records) - params.window
    if bootstrap_indices.shape[1] != evaluation_n:
        raise ValueError('Wrong bootstrap window')
    native = native_rows(records, params.window)
    keys = validated_pair_keys(native)
    ref = max(float(np.sqrt(np.mean(np.asarray(initial_context, float)**2))), params.min_scale)
    native_array = score_arrays(native, params.alpha)['winkler']
    result = {'native_q10_q90': {'metrics': interval_metrics(native, params.alpha),
                                'regime': {'high': None, 'other': None}},
              'signed_rolling_none': {}, 'expansion_only_rolling_none': 'none__rolling_cqr',
              'cells': {}, 'paired_vs_native': {}, 'paired_vs_none_rolling': {},
              'main_effects': {}, 'interactions': {}}
    per_cell = {}
    diagnostics_rows = []
    for scale in SCALES:
        for ctrl in CONTROLLERS:
            key = f'{scale}__{ctrl}'
            rows = calibrate(records, initial_context, params, scale, ctrl)
            if validated_pair_keys(rows) != keys:
                raise ValueError(f'Cell evaluation alignment drift: {key}')
            if len(rows) != evaluation_n:
                raise ValueError('Unexpected evaluation count')
            metrics = interval_metrics(rows, params.alpha)
            detail = diagnostics(rows, params, ref)
            result['cells'][key] = {'metrics': metrics, 'diagnostics': detail}
            per_cell[key] = score_arrays(rows, params.alpha)['winkler']
            for r, wink in zip(rows, per_cell[key]):
                diagnostics_rows.append({'cell': key, 'origin_index': r['origin_index'],
                                         'target_index': r['target_index'], 'origin_date': r['origin_date'],
                                         'target_date': r['target_date'], 'lower': r['lower'],
                                         'upper': r['upper'], 'target_log_return': r['target_log_return'],
                                         'winkler': float(wink), 'scale': r['scale'],
                                         'alpha_t': r['alpha_t'], 'gamma_t': r['gamma_t'],
                                         'gamma_driver': r['gamma_driver'],
                                         'return_volatility': r['return_volatility'],
                                         'residual_volatility': r['residual_volatility']})
    # A single causal, return-only gamma driver is required regardless of score scaling.
    for ctrl in CONTROLLERS:
        base = f'none__{ctrl}'
        baseline_rows = [r for r in diagnostics_rows if r['cell'] == base]
        driver = np.array([r['gamma_driver'] for r in baseline_rows])
        for scale in SCALES[1:]:
            other = [r for r in diagnostics_rows if r['cell'] == f'{scale}__{ctrl}']
            if not np.array_equal(driver, [r['gamma_driver'] for r in other]):
                raise ValueError('Gamma driver confounded by scaling')
    base = per_cell['none__rolling_cqr']
    for cell, a in per_cell.items():
        result['paired_vs_native'][cell] = paired_ci(a - native_array, bootstrap_indices)
        result['paired_vs_none_rolling'][cell] = paired_ci(a - base, bootstrap_indices)
    for scale in SCALES[1:]:
        effect = np.mean([per_cell[f'{scale}__{c}'] for c in CONTROLLERS], axis=0) - np.mean(
            [per_cell[f'none__{c}'] for c in CONTROLLERS], axis=0)
        result['main_effects'][f'{scale}_minus_none_mean_over_controllers'] = paired_ci(effect, bootstrap_indices)
    for ctrl in CONTROLLERS[1:]:
        effect = np.mean([per_cell[f'{s}__{ctrl}'] for s in SCALES], axis=0) - np.mean(
            [per_cell[f'{s}__rolling_cqr'] for s in SCALES], axis=0)
        result['main_effects'][f'{ctrl}_minus_rolling_mean_over_scales'] = paired_ci(effect, bootstrap_indices)
    for scale in SCALES[1:]:
        for ctrl in CONTROLLERS[1:]:
            interaction = ((per_cell[f'{scale}__{ctrl}'] - per_cell[f'{scale}__rolling_cqr'])
                           - (per_cell[f'none__{ctrl}'] - per_cell['none__rolling_cqr']))
            result['interactions'][f'{scale}__{ctrl}_vs_none_rolling'] = paired_ci(interaction, bootstrap_indices)
    signed = signed_rolling_diagnostic(records, params.window, params.alpha)
    signed_rows = signed.pop('rows')
    if signed_rows is not None:
        result['paired_vs_native']['signed_rolling_none'] = paired_ci(
            score_arrays(signed_rows, params.alpha)['winkler'] - native_array, bootstrap_indices)
        result['paired_vs_none_rolling']['signed_rolling_none'] = paired_ci(
            score_arrays(signed_rows, params.alpha)['winkler'] - base, bootstrap_indices)
    result['signed_rolling_none'] = signed
    # Raw baseline is evaluated on precisely the same 1006 origins.
    regime_mask = np.array([r['return_volatility'] > ref for r in
                            [r for r in diagnostics_rows if r['cell'] == 'none__rolling_cqr']])
    for name, mask in [('high', regime_mask), ('other', ~regime_mask)]:
        selected = [r for r, keep in zip(native, mask) if keep]
        result['native_q10_q90']['regime'][name] = {
            'n': len(selected), 'metrics': interval_metrics(selected, params.alpha) if selected else None}
    return result, diagnostics_rows
