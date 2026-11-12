"""TASK-005B offline tests. All synthetic values are illustrative, not SPY results."""
import numpy as np
import pytest

from cita2027.calibration.online import Parameters
from cita2027.calibration.development_analysis import (
    circular_block_indices, paired_ci, validated_pair_keys, signed_rolling_diagnostic,
    native_rows, score_arrays, diagnostics,
)


def sample_records(n=260, width=.02):
    return [{'origin_index': 3272 + i, 'target_index': 3273 + i,
             'origin_date': f'd{i}', 'target_date': f'd{i+1}',
             'q10': -.5 * width, 'q50': 0.0, 'q90': .5 * width,
             'target_log_return': .0, 'seed': 2027 + 3272 + i}
            for i in range(n)]


def test_bootstrap_deterministic_and_contiguous_blocks():
    a = circular_block_indices(1006, 100, 20, 2027)
    b = circular_block_indices(1006, 100, 20, 2027)
    assert np.array_equal(a, b) and a.shape == (100, 1006)
    assert np.all((a >= 0) & (a < 1006))
    for row in a:
        for j in range(0, 980, 20):
            assert np.array_equal(row[j:j+20], (row[j] + np.arange(20)) % 1006)


def test_paired_difference_identically_zero():
    i = circular_block_indices(100, 20, 10, 1)
    r = paired_ci(np.zeros(100), i)
    assert r['mean_difference'] == 0 and r['ci_95_percentile'] == [0, 0]
    assert r['ci_excludes_zero'] is False


def test_paired_constant_difference_sign():
    i = circular_block_indices(100, 20, 10, 1)
    r = paired_ci(np.full(100, -.01), i)
    assert r['ci_excludes_zero'] is True
    assert r['mean_difference'] == pytest.approx(-.01)


def test_duplicate_and_misaligned_keys_rejected():
    keys = native_rows(sample_records(), 252)
    assert len(validated_pair_keys(keys)) == 8
    keys[3]['origin_index'] = keys[2]['origin_index']
    with pytest.raises(ValueError):
        validated_pair_keys(keys)


def test_signed_cqr_can_contract_without_clipping():
    records = sample_records()
    result = signed_rolling_diagnostic(records, 252, .2)
    assert result['status'] == 'FINITE_SIGNED_INTERVALS'
    assert result['empty_count'] == 0
    assert result['metrics']['width_mean'] == pytest.approx(0)


def test_signed_empty_interval_detected_not_dropped():
    records = sample_records()
    records[-1]['q10'] = -.0001
    records[-1]['q90'] = .0001
    result = signed_rolling_diagnostic(records, 252, .2)
    assert result['status'] == 'EMPTY_CONFORMAL_SET_WINKLER_UNDEFINED'
    assert result['metrics'] is None
    assert result['empty_count'] == 1
    assert result['evaluated'] == 8
    assert result['coverage_with_empty_as_miss'] == 7 / 8


def test_raw_baseline_starts_after_identical_burn_in():
    raw = native_rows(sample_records(), 252)
    assert len(raw) == 8 and raw[0]['origin_index'] == 3524
    scores = score_arrays(raw, .2)
    assert np.allclose(scores['width'], .02)
    assert np.all(scores['covered'] == 1)


def test_bootstrap_invalid_alignment_rejected():
    indices = circular_block_indices(100, 10, 10)
    with pytest.raises(ValueError):
        paired_ci(np.ones(99), indices)


def test_diagnostics_are_not_conditional_coverage():
    p = Parameters()
    rows = [{'alpha_t': .2, 'gamma_t': 0.01, 'miss': 0, 'scale': 1.,
             'gamma_driver': 1., 'return_volatility': .1,
             'target_log_return': 0., 'lower': -.1, 'upper': .1}]
    d = diagnostics(rows, p, .05)
    assert d['volatility_regimes']['conditional_coverage_guarantee'] is False
    assert d['volatility_regimes']['high']['n'] == 1


def test_invalid_block_size_rejected():
    with pytest.raises(ValueError):
        circular_block_indices(100, 10, 101)


def test_full_factorial_cpu_end_to_end_synthetic_not_spy():
    from cita2027.calibration.development_analysis import analyze_model
    # 1,258 contiguous, finite forecast records (all toy/illustrative).
    records = sample_records(n=1258)
    initial = np.full(256, .005)
    res, csv_rows = analyze_model(records, initial, Parameters(),
                                  circular_block_indices(1006, reps=12, length=20))
    assert len(res['cells']) == 9
    assert len(csv_rows) == 9 * 1006
    assert len(res['paired_vs_native']) == 10  # 9 cells plus valid signed-CQR.
    assert res['signed_rolling_none']['empty_count'] == 0
    assert all(v['metrics']['n'] == 1006 for v in res['cells'].values())
    assert len(res['main_effects']) == 4
    assert len(res['interactions']) == 4
