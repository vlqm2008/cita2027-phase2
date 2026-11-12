"""TASK-006A focused CPU tests; no GPU or previously accepted audit reruns."""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from cita2027.validation.boundary import (
    DEV_PLAN_SHA, PROCESSED_SHA, SOURCE_ID, make_validation_plan,
    validation_window, validation_indices, check_boundary_records,
)
from cita2027.calibration.online import Parameters, calibrate


def synthetic_frame():
    # Dates bracket the real Development/Validation split but are synthetic sessions.
    d = list(pd.bdate_range(end='2022-12-30', periods=4531))
    d += list(pd.bdate_range(start='2023-01-03', periods=1000))
    n = len(d)
    r = np.linspace(-0.005, 0.005, n)
    out = pd.DataFrame({'date': d, 'split': ['warmup'] * 3272 + ['development'] * 1259 +
                        ['validation'] * 502 + ['final_test'] * (n - 5033),
                        'log_return': r})
    out.loc[0, 'log_return'] = np.nan
    out['previous_date'] = out['date'].shift(1)
    out['previous_row_index'] = pd.Series(np.arange(n)).shift(1).astype('Int64')
    return out


def dev_stub():
    return {'count': 1258, 'last': {'origin_index': 4529, 'target_index': 4530},
            'processed_sha256': PROCESSED_SHA,
            'checkpoint_audit_sha256': 'abc', 'source_snapshot_id': SOURCE_ID,
            'smoke_config_sha256': 'smoke', 'seed': 2027,
            'context_length': 256, 'horizon': 1, 'samples': 100,
            'quantile_levels': [0.1, 0.5, 0.9]}


def test_bridge_plan_count_and_alignment():
    df = synthetic_frame()
    plan = make_validation_plan(df, dev_stub(), 'abc')
    assert plan['count'] == 502 and plan['bridge_count'] == 1
    assert (plan['first']['origin_index'], plan['first']['target_index']) == (4530, 4531)
    assert (plan['last']['origin_index'], plan['last']['target_index']) == (5031, 5032)
    assert plan['first']['origin_date'] == '2022-12-30'
    assert plan['first']['target_date'] == '2023-01-03'
    assert not any(row['target_index'] >= 5033 for row in plan['origins'])
    assert plan['origins'][0]['role'] == 'boundary_bridge'
    assert all(row['seed'] == 2027 + row['origin_index'] for row in plan['origins'])


def test_target_perturbation_cannot_change_forecast_context():
    df = synthetic_frame()
    before = validation_window(df, 4530)
    changed = df.copy()
    changed.loc[4531, 'log_return'] = 999.0
    after = validation_window(changed, 4530)
    assert before.context_sha256 == after.context_sha256
    assert np.array_equal(before.context, after.context)
    assert before.target_log_return != after.target_log_return


def test_bridge_context_shifts_one_session():
    df = synthetic_frame()
    first, second = validation_window(df, 4530), validation_window(df, 4531)
    np.testing.assert_equal(first.context[1:], second.context[:-1])
    assert first.target_log_return == second.context[-1]


def test_wrong_split_or_future_test_rejected():
    df = synthetic_frame()
    with pytest.raises(ValueError):
        validation_window(df, 5032)  # final target is forbidden
    corrupted = df.copy()
    corrupted.loc[4531, 'split'] = 'development'
    with pytest.raises(ValueError):
        validation_indices(corrupted)
    with pytest.raises(ValueError):
        validation_window(corrupted, 4530)


def test_nonadjacent_calendar_rejected():
    df = synthetic_frame()
    df.loc[4531, 'previous_row_index'] = 4529
    with pytest.raises(ValueError):
        validation_window(df, 4530)


def fake_records():
    rows = []
    for i in range(3272, 5032):
        rows.append({'origin_index': i, 'target_index': i + 1,
                     'origin_date': str(i), 'target_date': str(i + 1),
                     'seed': 2027 + i, 'q10': -0.015, 'q50': 0.0,
                     'q90': 0.015, 'target_log_return': 0.002 * np.cos(i / 19)})
    return rows


def test_state_transfer_identical_to_one_continuous_stream():
    rows = fake_records()
    initial = np.linspace(-.007, .007, 256)
    p = Parameters()
    for scale, controller in [('none', 'rolling_cqr'),
                              ('return_ewma', 'fixed_gamma_aci'),
                              ('residual_ewma', 'variable_gamma_aci')]:
        whole = calibrate(rows, initial, p, scale, controller)
        assert len(whole) == 1508
        assert whole[1006]['origin_index'] == 4530
        assert whole[1006]['target_index'] == 4531
        assert all(not r['burn_in'] for r in whole[-502:])
        assert np.isfinite([r['lower'] for r in whole[-502:]]).all()
        assert np.isfinite([r['upper'] for r in whole[-502:]]).all()


def test_future_target_change_only_affects_next_origin_state():
    rows = fake_records()
    changed = copy.deepcopy(rows)
    changed[1258]['target_log_return'] += .15
    initial = np.linspace(-.007, .007, 256)
    base = calibrate(rows, initial, Parameters(), 'return_ewma', 'variable_gamma_aci')
    alt = calibrate(changed, initial, Parameters(), 'return_ewma', 'variable_gamma_aci')
    # At bridge origin, prediction interval/state must NOT use its future target.
    for key in ('lower', 'upper', 'alpha_t', 'scale', 'gamma_t'):
        assert base[1006][key] == alt[1006][key]
    assert base[1007]['scale'] != alt[1007]['scale']


def test_missing_bridge_rejected():
    rows = fake_records()
    df = synthetic_frame()
    plan = make_validation_plan(df, dev_stub(), 'abc')
    fake = rows[:1258], rows[1259:]
    with pytest.raises(ValueError):
        check_boundary_records(fake[0], fake[1], dev_stub(), plan)


def test_regime_threshold_at_bridge_cannot_see_bridge_target():
    from cita2027.calibration.diagnostic_correction import causal_regime_indices
    rows = fake_records()
    mutated = copy.deepcopy(rows)
    mutated[1258]['target_log_return'] = 0.25
    initial = np.linspace(-0.007, 0.007, 256)
    a = causal_regime_indices(rows, initial)
    b = causal_regime_indices(mutated, initial)
    j = 1258 - 252
    assert a['sigma'][j] == b['sigma'][j]
    assert a['threshold'][j] == b['threshold'][j]
    assert a['high'][j] == b['high'][j]
    assert a['sigma'][j + 1] != b['sigma'][j + 1]
