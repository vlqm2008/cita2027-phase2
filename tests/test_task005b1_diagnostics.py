"""Focused synthetic tests for TASK-005B.1; not SPY empirical evidence."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pytest

from cita2027.calibration.diagnostic_correction import (
    causal_regime_indices, masked_bootstrap_ci, gamma_diagnostics,
    load_traces, validate_input_report, BOOTSTRAP_SEED,
)
from cita2027.calibration.development_analysis import circular_block_indices
from cita2027.calibration.online import Parameters


def toy(n=260):
    return [{'origin_index': 3272+i, 'target_index': 3273+i,
             'seed': 2027 + 3272+i, 'target_log_return': .01 if i < 255 else .09}
            for i in range(n)]


def test_regime_causal_future_target_perturbation():
    r = toy()
    a = causal_regime_indices(r, np.full(256, .01), window=252)
    r[252]['target_log_return'] = -.45
    b = causal_regime_indices(r, np.full(256, .01), window=252)
    assert a['sigma'][0] == b['sigma'][0]
    assert a['threshold'][0] == b['threshold'][0]
    assert bool(a['high'][0]) == bool(b['high'][0])
    assert a['sigma'][1] != b['sigma'][1]


def test_regime_threshold_only_prior_volatility():
    r = toy()
    a = causal_regime_indices(r, np.full(256, .01), window=252)
    r[-1]['target_log_return'] = 50.
    b = causal_regime_indices(r, np.full(256, .01), window=252)
    assert np.array_equal(a['threshold'], b['threshold'])
    assert np.array_equal(a['high'], b['high'])


def test_regime_warmup_and_index_guards():
    with pytest.raises(ValueError, match='Invalid causal regime'):
        causal_regime_indices(toy(250), np.full(256, .01))
    r = toy(); r[13]['seed'] = 99
    with pytest.raises(ValueError, match='Index/seed'):
        causal_regime_indices(r, np.full(256, .01))


def test_ties_classified_other():
    r = toy(); r[:] = [{**x, 'target_log_return': .01} for x in r]
    output = causal_regime_indices(r, np.full(256, .01))
    assert not output['high'].any()
    assert np.allclose(output['sigma'], output['threshold'])


def test_masked_bootstrap_determinism_and_no_group():
    a = np.arange(30, dtype=float)
    mask = a > 20
    draw = circular_block_indices(30, 80, 10, 11)
    x = masked_bootstrap_ci(a, mask, draw)
    assert x == masked_bootstrap_ci(a, mask, draw)
    assert x['n'] == 9 and x['mean'] == 25
    assert masked_bootstrap_ci(a, np.zeros(30, dtype=bool), draw)['mean'] is None


def test_gamma_saturation_and_effective_gamma():
    d = pd.DataFrame({'cell': ['none__variable_gamma_aci'] * 4,
                      'gamma_driver': [.5, 1, 2, 2],
                      'gamma_t': [.005, .01, .02, .02],
                      'origin_date': ['2019-01-01']*4})
    res = gamma_diagnostics(d, Parameters())
    assert res['driver_at_max_fraction'] == .5
    assert res['driver_at_min_fraction'] == .25
    assert res['effective_gamma_distribution']['max'] == .02
    d.loc[1, 'gamma_t'] = .2
    with pytest.raises(ValueError, match='Effective gamma'):
        gamma_diagnostics(d, Parameters())


def test_cross_model_paired_block_bootstrap_not_independent():
    d = np.arange(40, dtype=float) - 20
    draw = circular_block_indices(40, 40, 10, BOOTSTRAP_SEED)
    a = masked_bootstrap_ci(d, np.ones(40, bool), draw)
    b = masked_bootstrap_ci(-d, np.ones(40, bool), draw)
    assert np.allclose(a['ci_95'], [-b['ci_95'][1], -b['ci_95'][0]])


def test_fixed_gamma_unchanged_base_parameters():
    from dataclasses import replace
    p = Parameters()
    fixed = replace(p, gamma=.02)
    assert p.gamma == .01 and fixed.gamma == .02 and fixed.window == 252


def test_tampered_report_rejected(tmp_path):
    path = tmp_path / 'report.json'
    path.write_text('{"status": "PASS"}')
    with pytest.raises(ValueError, match='SHA'):
        validate_input_report(path)


def test_tampered_trace_rejected(tmp_path):
    p = tmp_path / 'trace.csv'
    p.write_text('bad')
    with pytest.raises(ValueError, match='SHA'):
        load_traces(p, {'diagnostic_trace_sha256': '0' * 64})


def test_explanatory_end_to_end_synthetic_1258_origins():
    """CPU integration with synthetic returns, not a claim about SPY observations."""
    from cita2027.calibration.online import calibrate, SCALES, CONTROLLERS
    from cita2027.calibration.development_analysis import score_arrays
    from cita2027.calibration.diagnostic_correction import evaluate_report
    i = np.arange(1258)
    y = 0.004 * np.sin(i / 7) + np.where((i // 70) % 2 == 0, .006, .015) * np.cos(i / 3)
    rows = [{'origin_index': int(3272 + j), 'target_index': int(3273 + j),
             'seed': int(2027 + 3272 + j), 'target_log_return': float(y[j]),
             'q10': -.01, 'q50': 0., 'q90': .01,
             'origin_date': str(pd.Timestamp('2018-01-01') + pd.Timedelta(days=j)),
             'target_date': str(pd.Timestamp('2018-01-02') + pd.Timedelta(days=j))}
            for j in range(1258)]
    ctx = np.full(256, .01)
    frames = {}
    for model in ('chronos', 'moirai'):
        cells = {}
        for scale in SCALES:
            for ctrl in CONTROLLERS:
                key = f'{scale}__{ctrl}'
                d = pd.DataFrame(calibrate(rows, ctx, Parameters(), scale, ctrl))
                d['cell'] = key
                d['winkler'] = score_arrays(d.to_dict('records'), .2)['winkler']
                cells[key] = d
        frames[model] = cells
    res = evaluate_report({}, frames, {'chronos': rows, 'moirai': rows}, ctx,
                          bootstrap_reps=12)
    assert sum(res['regime_counts'].values()) == 1006
    assert all(res['regime_counts'][k] > 0 for k in ('high', 'other'))
    assert len(res['models']['chronos']['fixed_gamma_0_02_sensitivity']) == 9
    assert len(res['models']['moirai']['direct_paired_contrasts']) == 15
