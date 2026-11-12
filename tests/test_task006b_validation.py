"""TASK-006B tests: only boundary guards, Validation report structural safety."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from task006b_validation import CELLS, DEV_PLAN_SHA, summarize, verify_locked_plan, verify_report, verify_development_anchors
from cita2027.forecasting.cache import digest
from cita2027.utils.provenance import canonical_json_bytes


def fake_plan():
    # The unit test explicitly verifies rejection of nonlocked synthetic plan.
    return {'development_plan_sha256': DEV_PLAN_SHA, 'count': 502,
            'origins': [{'origin_index': i, 'target_index': i + 1, 'seed': 2027+i}
                        for i in range(4530, 5032)]}


def fake_report():
    metric = {'n': 502, 'winkler_mean': 0.05, 'coverage': .8,
              'width_mean': .03, 'lower_miss_rate': .12, 'upper_miss_rate': .08}
    reg = {'high': {'n': 120, 'valid_bootstrap_replicates': 1000},
           'other': {'n': 382, 'valid_bootstrap_replicates': 1000}}
    stat = {str(L): {'mean': -.001, 'ci95': [-.002,.003], 'exploratory': True,
                     'multiplicity_adjusted': False} for L in (10,20,40)}
    cell = {'metrics': metric, 'regimes': reg,
            'paired_contrasts': {'unscaled_rolling': stat, 'same_scale_fixed': stat}}
    methods = {k: copy.deepcopy(cell) for k in CELLS}
    methods['raw_q10_q90'] = {'metrics': metric}
    methods['signed_rolling_none'] = {'status': 'FINITE_SIGNED_INTERVALS',
                                      'metrics': metric, 'empty_origins_in_validation': []}
    return {'validation_plan_sha256': digest(fake_plan()), 'development_plan_sha256': DEV_PLAN_SHA,
            'records_per_model': 502, 'final_test_used': False,
            'method_selection_performed': False, 'validation_hyperparameter_tuning': False,
            'contamination_free_verified': False, 'point_in_time_adjusted_prices_verified': False,
            'scientific_clearance': False, 'regime_counts': {'high': 120, 'other': 382},
            'models': {'chronos': copy.deepcopy(methods), 'moirai': copy.deepcopy(methods)}}


def test_plan_gate_rejects_unpublished_or_different_sha(tmp_path):
    with pytest.raises(ValueError, match='immutable Validation plan'):
        verify_locked_plan(tmp_path, fake_plan())


def test_plan_gate_rejects_wrong_exact_sha_even_if_file_matches(tmp_path):
    plan = fake_plan()
    sha = digest(plan)
    path = tmp_path / 'artifacts/forecast_cache/validation/plans' / f'{sha}.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical_json_bytes(plan))
    with pytest.raises(ValueError, match='immutable Validation plan'):
        verify_locked_plan(tmp_path, plan, expected_sha=sha)


def test_report_rejects_future_test_usage():
    report = fake_report()
    report['final_test_used'] = True
    with pytest.raises(ValueError, match='unsafe Validation'):
        verify_report(report, fake_plan())


def test_report_rejects_validation_tuning():
    report = fake_report()
    report['validation_hyperparameter_tuning'] = True
    with pytest.raises(ValueError, match='unsafe Validation'):
        verify_report(report, fake_plan())


def test_report_rejects_origin_count_mismatch():
    report = fake_report()
    report['models']['moirai'][CELLS[0]]['metrics']['n'] = 501
    with pytest.raises(ValueError, match='invalid metrics'):
        verify_report(report, fake_plan())


def test_report_rejects_empty_regime():
    report = fake_report()
    report['regime_counts'] = {'high':0, 'other':502}
    with pytest.raises(ValueError, match='volatility regime'):
        verify_report(report, fake_plan())


def test_report_rejects_missing_block_sensitivity():
    report = fake_report()
    del report['models']['chronos'][CELLS[0]]['paired_contrasts']['same_scale_fixed']['40']
    with pytest.raises(ValueError, match='bootstrap length'):
        verify_report(report, fake_plan())


def test_report_rejects_signed_cqr_empty_sets_mishandled():
    report = fake_report()
    report['models']['chronos']['signed_rolling_none']['status'] = 'VALIDATION_EMPTY_SIGNED_SET'
    with pytest.raises(ValueError, match='empty-set'):
        verify_report(report, fake_plan())


def test_report_rejects_miss_coverage_incoherence():
    report = fake_report()
    report['models']['chronos'][CELLS[0]]['metrics']['lower_miss_rate'] = .3
    with pytest.raises(ValueError, match='invalid metrics'):
        verify_report(report, fake_plan())


def test_summary_never_selects_winner_or_tunes():
    report = fake_report()
    plan = fake_plan()
    report['development_plan_sha256'] = plan['development_plan_sha256']
    report['validation_plan_sha256'] = digest(plan)
    dev_metrics = {'n': 1006, 'winkler_mean': 0.06, 'coverage': .79}
    dev_model = {'cells': {cell: {'metrics': dev_metrics} for cell in CELLS},
                 'native_q10_q90': {'metrics': dev_metrics}}
    dev = {'full_plan_sha256': plan['development_plan_sha256'],
           'models': {'chronos': dev_model, 'moirai': dev_model}}
    out = summarize(report, dev, plan, 'sha-val', {'chronos': {}, 'moirai': {}})
    assert out['selection_performed'] is False
    assert out['scientific_clearance'] is False
    assert set(out['models']['chronos']['all_nine_factorial_cells']) == set(CELLS)
    assert out['models']['chronos']['all_nine_factorial_cells'][CELLS[0]]['validation_minus_development_winkler'] == pytest.approx(-.01)


def test_preflight_rejects_missing_development_report(tmp_path):
    with pytest.raises(ValueError, match='Development calibration report drift'):
        verify_development_anchors(tmp_path)
