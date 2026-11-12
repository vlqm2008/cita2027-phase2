import numpy as np
import pytest

from cita2027.calibration.online import (Parameters, calibrate, signed_score,
    conformal_order_statistic, interval, interval_metrics, SCALES, CONTROLLERS)


def sample(n=310, shocked=False):
    rows=[]
    for j in range(n):
        i=3272+j
        # Pure deterministic toy series; base intervals are ordered.
        y=(0.005*np.sin(j*0.4) + (0.025 if shocked and j==280 else 0))
        rows.append(dict(origin_index=i,target_index=i+1,seed=2027+i,
                         origin_date=f"day-{j}",target_date=f"day-{j+1}",
                         q10=-0.008,q50=0.0,q90=0.008,target_log_return=y))
    return rows


def test_signed_score_and_boundary_ties():
    assert signed_score(0,-1,1)==-1
    assert signed_score(-2,-1,1)==1
    assert signed_score(1,-1,1)==0
    assert interval(-1,1,-0.5,1,'signed_interval')==(-0.5,0.5)
    assert interval(-1,1,-0.5,1,'expansion_only')==(-1,1)
    with pytest.raises(ValueError, match='empty'):
        interval(-0.01,0.01,-0.5,1,'signed_interval')
    assert interval(-1,1,0.2,2)==(-1.4,1.4)


def test_conformal_rank_ties_infinity_and_no_interpolation():
    assert conformal_order_statistic([1,2,3,4],.5)==3
    assert conformal_order_statistic([2,2,2,9],.5)==2
    assert np.isinf(conformal_order_statistic([1,2,3,4],.1))
    with pytest.raises(ValueError):
        conformal_order_statistic([float('nan')],.2)


def test_lookahead_perturbation_changes_only_future():
    a=sample(); b=sample(shocked=True)
    p=Parameters(window=252)
    for scale in SCALES:
        for ctl in CONTROLLERS:
            x=calibrate(a,np.full(256,.01),p,scale,ctl)
            y=calibrate(b,np.full(256,.01),p,scale,ctl)
            assert x[28]['origin_index']==3552
            # Target at j=280 is first observed AFTER that interval is generated.
            assert x[:28]==y[:28]
            assert (x[28]['lower'],x[28]['upper'],x[28]['alpha_t'],x[28]['scale']) == (y[28]['lower'],y[28]['upper'],y[28]['alpha_t'],y[28]['scale'])
            # Next day's residual/return scales may change; neither can alter past.
            if scale!='none':
                assert x[29]['scale']!=y[29]['scale']


def test_aci_only_updates_after_current_target():
    p=Parameters(window=252,gamma=.03)
    out=calibrate(sample(), np.full(256,.01), p,'none','fixed_gamma_aci')
    assert out[0]['alpha_t']==.2
    for prev, nxt in zip(out,out[1:]):
        expected=np.clip(prev['alpha_t']+.03*(.2-prev['miss']),1/253,1-1/253)
        assert nxt['alpha_t']==pytest.approx(expected)


def test_alpha_boundaries_and_no_infinite_interval():
    p=Parameters(window=252,gamma=10)
    out=calibrate(sample(),np.full(256,.01),p,'none','fixed_gamma_aci')
    assert all(1/253<=x['alpha_t']<=1-1/253 for x in out)
    assert all(np.isfinite(x['lower']) and np.isfinite(x['upper']) for x in out)


def test_scale_ewma_recurrence():
    p=Parameters(window=252,decay=.94)
    arr=sample()
    arr[0]['target_log_return']=.02
    for scale in ('residual_ewma','return_ewma'):
        out=calibrate(arr,np.full(256,.01),p,scale,'rolling_cqr')
        expected=.94*.0001+.06*(.02**2)
        # Later updates differ but all scales are positive and finite.
        assert expected>0
        assert all(x['scale']>0 and np.isfinite(x['scale']) for x in out)


def test_gamma_driver_identical_across_scales():
    p=Parameters(window=252)
    results=[calibrate(sample(),np.full(256,.01),p,s,'variable_gamma_aci') for s in SCALES]
    for j in range(len(results[0])):
        assert len({r[j]['gamma_driver'] for r in results})==1
        assert len({r[j]['gamma_t'] for r in results})==1


def test_determinism_and_rolling_buffer():
    p=Parameters(window=252)
    a=calibrate(sample(),np.full(256,.01),p,'return_ewma','rolling_cqr')
    b=calibrate(sample(),np.full(256,.01),p,'return_ewma','rolling_cqr')
    assert a==b and len(a)==58
    assert a[0]['origin_index']==3272+252


def test_winkler_and_tail_decomposition():
    rows=[dict(lower=-1,upper=1,target_log_return=0),
          dict(lower=-1,upper=1,target_log_return=2),
          dict(lower=-1,upper=1,target_log_return=-2)]
    x=interval_metrics(rows,.2,rolling_window=2)
    assert x['coverage']==pytest.approx(1/3)
    assert x['lower_miss_rate']==pytest.approx(1/3)
    assert x['upper_miss_rate']==pytest.approx(1/3)
    assert x['winkler_mean']==pytest.approx((2+12+12)/3)
    assert x['conditional_coverage'] is None


def test_reject_out_of_order_duplicate_and_bad_seed():
    p=Parameters()
    rows=sample()
    rows[290]['origin_index']=rows[289]['origin_index']
    with pytest.raises(ValueError):
        calibrate(rows,np.full(256,.01),p,'none','rolling_cqr')
    rows=sample()
    rows[0]['seed']=3
    with pytest.raises(ValueError):
        calibrate(rows,np.full(256,.01),p,'none','rolling_cqr')


def test_invalid_params_and_context():
    with pytest.raises(ValueError):
        Parameters(decay=1.0)
    with pytest.raises(ValueError):
        calibrate(sample(), np.ones(255), Parameters(), 'none','rolling_cqr')


def test_ewma_exact_update_at_next_origin():
    params = Parameters(window=2, alpha=.5, decay=.9, min_scale=1e-9)
    rows = sample(4)
    # Initialize return and residual variance from constant context RMS 0.01.
    context = np.full(256, .01)
    rows[0]['target_log_return'] = .02
    rows[1]['target_log_return'] = -.01
    rows[2]['target_log_return'] = .005
    res = calibrate(rows, context, params, 'residual_ewma','rolling_cqr')
    ret = calibrate(rows, context, params, 'return_ewma','rolling_cqr')
    # At evaluation origin j=2, both saw exactly y_0 and y_1, no y_2.
    expected_v = .9*(.9*.0001+.1*.02**2)+.1*.01**2
    assert res[0]['scale'] == pytest.approx(np.sqrt(expected_v))
    assert ret[0]['scale'] == pytest.approx(np.sqrt(expected_v))
    assert res[0]['scale'] == pytest.approx(res[0]['return_volatility'])


def test_common_driver_independent_of_current_target_and_raw_quantiles():
    params = Parameters(window=252)
    rows_a, rows_b = sample(), sample()
    rows_b[280]['target_log_return'] = 50
    rows_b[280]['q50'] = .003
    a = calibrate(rows_a, np.full(256,.01), params, 'residual_ewma','variable_gamma_aci')
    b = calibrate(rows_b, np.full(256,.01), params, 'residual_ewma','variable_gamma_aci')
    for key in ('gamma_t','gamma_driver','alpha_t'):
        assert a[28][key] == b[28][key]
    assert a[28]['gamma_driver'] == b[28]['gamma_driver']
    assert a[29]['residual_volatility'] != b[29]['residual_volatility']


def test_equal_endpoints_are_covered_and_miss_is_strict():
    rows=[dict(lower=0.,upper=1.,target_log_return=0.),
          dict(lower=0.,upper=1.,target_log_return=1.)]
    met=interval_metrics(rows, .2)
    assert met['coverage']==1.0 and met['winkler_mean']==1.0
