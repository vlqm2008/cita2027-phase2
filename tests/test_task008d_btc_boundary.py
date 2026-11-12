"""Focused CPU-only regression tests. No market sources or GPU inference."""
from __future__ import annotations
from datetime import date,timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cita2027.btc.ingest import origin_plan, forecast_context
from cita2027.calibration.online import Parameters, calibrate
from task008d_btc_boundary import (boundary_plan_check, initial_state, one_transition,
    restore_state, state_payload, state_replay, regime_boundary, synthetic_probe_inputs,
    calendar_static_check, digest)

@pytest.fixture(scope='module')
def sample():
    days=pd.date_range('2017-08-17',periods=2694,freq='D')
    ix=np.arange(2694)
    log_ret=np.r_[np.nan,.014*np.sin(ix[1:]/9)+.009*np.cos(ix[1:]/17)]
    df=pd.DataFrame({'date':days.strftime('%Y-%m-%d'),'row_index':ix,
        'split':['warmup' if i<502 else 'development' if i<1963 else 'validation' for i in ix],
        'log_return':log_ret})
    dev=origin_plan(df,'05088f11c7929995ef6dd28e5135669ebaf37465928c62c826a15f245520299a','SYNTHETIC','development')
    val=origin_plan(df,'05088f11c7929995ef6dd28e5135669ebaf37465928c62c826a15f245520299a','SYNTHETIC','validation')
    rows=[{'origin_index':p['origin_index'],'target_index':p['target_index'],
        'origin_date':p['origin_date'],'target_date':p['target_date'],
        'seed':p['seed'],'q10':-.01,'q50':.0,'q90':.01,
        'target_log_return':float(df.iloc[p['target_index']]['log_return'])} for p in dev['origins']]
    return df,dev,val,rows

def test_bridge_includes_weekend(sample):
    df,dev,val,_=sample
    f=boundary_plan_check(dev,val,df.iloc[:1963])
    assert f['origin_index']==1962 and f['target_index']==1963 and f['seed']==3989
    assert f['context_first_index']==1707 and len(val['origins'])==731
    assert val['origins'][1]['target_date']=='2023-01-02'

def test_no_incorrect_weekday_skip(sample):
    df,dev,val,_=sample
    wrong=dict(val); wrong['origins']=[dict(x) for x in val['origins']]
    wrong['origins'][0]['target_date']='2023-01-02'
    with pytest.raises(ValueError):boundary_plan_check(dev,wrong,df.iloc[:1963])

def test_future_target_does_not_change_bridge_context(sample):
    df,dev,val,_=sample
    a=forecast_context(df,1962)
    changed=df.copy();changed.loc[1963,'log_return']=1e9
    assert np.array_equal(a,forecast_context(changed,1962))

def test_moirai_synthetic_index_guard():
    root=Path(__file__).resolve().parents[1]
    # Unit test file is run in local overlay, requiring copy of locked adapter.
    assert calendar_static_check(root)['bdays_are_not_actual_btc_target_dates'] is True

def test_replay_matches_original_and_state_persists(sample):
    df,dev,val,rows=sample
    p=Parameters(); context=forecast_context(df,501)
    states=state_replay(rows,context,p)
    assert len(states)==9
    for key,info in states.items():
        s=restore_state(info['state'],p)
        assert s['processed']==1461 and len(s['scores'])==252
        assert digest(state_payload(s))==info['state_sha256']

def test_no_reset_at_validation_and_first_three_state_probes(sample):
    df,dev,val,rows=sample;p=Parameters();first=forecast_context(df,501)
    states=state_replay(rows,first,p)
    probes=synthetic_probe_inputs(rows,val)
    assert [r['origin_index'] for r in probes]==[1962,1963,1964]
    for key,x in states.items():
        scale,control=key.split('__')
        carry=restore_state(x['state'],p)
        reset=initial_state(first,p)
        assert digest(state_payload(carry))!=digest(state_payload(reset))
        first_out=one_transition(carry,probes[0],p,scale,control,expect_output=True)
        assert first_out is not None and carry['processed']==1462
        assert len(carry['scores'])==252

def test_regime_uses_prior_origins_and_no_future(sample):
    df,dev,val,rows=sample; first=forecast_context(df,501)
    a=regime_boundary(rows,first,Parameters())
    changed=[dict(r) for r in rows]
    # Regime at 2022-12-31 -> 2023-01-01 must be invariant to any later val target.
    assert a==regime_boundary(changed,first,Parameters())
    assert a['prior_sigma_count']==252

def test_variable_gamma_next_alpha_bounds(sample):
    df,dev,val,rows=sample;first=forecast_context(df,501)
    state=state_replay(rows,first,Parameters())
    for k,v in state.items():
        if k.endswith('aci'):
            assert 1/253<=v['state']['alpha_next']<=252/253

def test_protocol_probe_matches_uninterrupted_calibrate(sample):
    df,dev,val,rows=sample
    p=Parameters(); ctx=forecast_context(df,501)
    states=state_replay(rows,ctx,p)
    probes=synthetic_probe_inputs(rows,val)
    for key,item in states.items():
        scale,control=key.split('__')
        expected=calibrate(rows+probes,ctx,p,scale,control)[-3:]
        carry=restore_state(item['state'],p)
        for k,r in enumerate(probes):
            got=one_transition(carry,r,p,scale,control,expect_output=True)
            for field in ('lower','upper','alpha_t','scale','gamma_driver','gamma_t','return_volatility','residual_volatility','miss'):
                assert np.isclose(got[field],expected[k][field],atol=1e-12,rtol=1e-12), (key,k,field)


def test_future_target_changes_feedback_but_not_pre_prediction_interval(sample):
    df,dev,val,rows=sample
    ctx=forecast_context(df,501);p=Parameters()
    state=state_replay(rows,ctx,p)['residual_ewma__variable_gamma_aci']['state']
    a=restore_state(state,p);b=restore_state(state,p)
    rec=synthetic_probe_inputs(rows,val,1)[0]
    changed={**rec,'target_log_return':float(rec['target_log_return'])+1.0}
    x=one_transition(a,rec,p,'residual_ewma','variable_gamma_aci',expect_output=True)
    y=one_transition(b,changed,p,'residual_ewma','variable_gamma_aci',expect_output=True)
    for k in ('lower','upper','alpha_t','scale','gamma_driver','gamma_t','return_volatility','residual_volatility'):
        assert x[k]==y[k]
    assert digest(state_payload(a))!=digest(state_payload(b))
