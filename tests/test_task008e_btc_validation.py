"""Offline, CPU-only tests for validation cache and causal replay."""
from __future__ import annotations
from datetime import date,timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from cita2027.btc.validation import (
    DATA_SHA,DEV_SHA,VAL_SHA,VAL_COUNT,FIRST_ORIGIN,check_record,cache_records,
    sanity_metrics,SCHEMA,inputs,val_cache,
)
from cita2027.btc.ingest import origin_plan,forecast_context
from cita2027.calibration.online import Parameters,calibrate
from cita2027.forecasting.cache import write_once_atomic
from cita2027.utils.provenance import canonical_json_bytes

@pytest.fixture(scope='module')
def snapshot():
    n=2694
    day=pd.date_range('2017-08-17',periods=n,freq='D')
    x=np.arange(n,dtype=float)
    closes=100*np.exp(0.0004*x+0.03*np.sin(x/21))
    returns=np.r_[np.nan,np.log(closes[1:]/closes[:-1])]
    split=np.select([x<502,x<1963],['warmup','development'],default='validation')
    return pd.DataFrame({'row_index':np.arange(n),'date':day.strftime('%Y-%m-%d'),
                         'split':split,'log_return':returns})

@pytest.fixture(scope='module')
def plan(snapshot):return origin_plan(snapshot,DATA_SHA,'TEST_SNAPSHOT','validation')

def fake_record(snapshot,p,model='chronos'):
    return {'schema_version':SCHEMA,'model':model,'plan_sha256':VAL_SHA,
     'dataset_sha256':DATA_SHA,'source_manifest_sha256':'76bbda3bef3be823324f237b5f95354ea55b0e75085a5d9d52b64351fe5f0f5f',
     'checkpoint_id':'mock','checkpoint_commit':'rev','checkpoint_weight_sha256':'weight',
     'origin_index':p['origin_index'],'target_index':p['target_index'],
     'origin_date':p['origin_date'],'target_date':p['target_date'],
     'context_first_index':p['context_first_index'],'context_last_index':p['context_last_index'],
     'context_sha256':p['context_sha256'],'seed':p['seed'],'context_length':256,
     'horizon':1,'samples':100,'quantile_levels':[.1,.5,.9],
     'target_units':'daily_log_return_decimal','q10':-.02,'q50':0.,'q90':.02,
     'target_log_return':float(snapshot.iloc[p['target_index']]['log_return']),
     'elapsed_seconds':0.1,'peak_allocated_mib':10.,
     'replay_checked':p['origin_index'] in (1962,1963),'replay_max_abs_diff':0.,
     'contamination_free_verified':False,'btc_historical_point_in_time_verified':False,'scientific_clearance':False}

def identity():return (None,'mock','rev','weight',{})

def test_validation_count_weekend_bridge(plan):
    assert len(plan['origins'])==VAL_COUNT
    p=plan['origins'][0];assert (p['origin_index'],p['target_index'])==(1962,1963)
    assert p['context_first_index']==1707 and p['seed']==3989
    assert (p['origin_date'],p['target_date'])==('2022-12-31','2023-01-01')
    assert plan['origins'][-1]['target_date']=='2024-12-31'
    assert plan['origins'][6]['target_date']=='2023-01-07'

def test_valid_forecast_record(snapshot,plan):
    p=plan['origins'][0]
    check_record(fake_record(snapshot,p),p,'chronos',identity(),snapshot)

def test_wrong_seed_rejected(snapshot,plan):
    p=plan['origins'][0];r=fake_record(snapshot,p);r['seed']=2027
    with pytest.raises(ValueError,match='METADATA'):check_record(r,p,'chronos',identity(),snapshot)

def test_future_target_never_in_context(snapshot,plan):
    i=plan['origins'][0]['origin_index']
    before=forecast_context(snapshot,i)
    other=snapshot.copy();other.loc[i+1,'log_return']=999.
    assert np.array_equal(before,forecast_context(other,i))
    other.loc[i,'log_return']+=1.
    assert not np.array_equal(before,forecast_context(other,i))

def test_target_mutation_rejected(snapshot,plan):
    p=plan['origins'][0];r=fake_record(snapshot,p);r['target_log_return']+=.0001
    with pytest.raises(ValueError,match='TARGET_MISMATCH'):check_record(r,p,'chronos',identity(),snapshot)

def test_quantile_inversion_rejected(snapshot,plan):
    p=plan['origins'][0];r=fake_record(snapshot,p);r['q10']=.1
    with pytest.raises(ValueError):check_record(r,p,'chronos',identity(),snapshot)

def test_noncontiguous_or_unplanned_records_fail(snapshot,plan,tmp_path):
    p=plan['origins'][1];r=fake_record(snapshot,p)
    f=val_cache(tmp_path,'chronos')/'records'/f'{p["origin_index"]:06d}.json'
    write_once_atomic(f,canonical_json_bytes(r))
    with pytest.raises(ValueError,match='NONCONTIGUOUS'):
        cache_records(tmp_path,'chronos',plan,identity(),snapshot)

def test_write_once_fails_drift(tmp_path):
    f=tmp_path/'x.json';write_once_atomic(f,b'abc')
    with pytest.raises(Exception):write_once_atomic(f,b'xyz')

def test_carry_forward_no_reset_in_calibrate():
    params=Parameters()
    initial=np.linspace(-.02,.02,256)
    stream=[]
    for j in range(300):
        origin=501+j
        stream.append({'origin_index':origin,'target_index':origin+1,'seed':2027+origin,
            'origin_date':str(origin),'target_date':str(origin+1),
            'q10':-.01,'q50':0.,'q90':.01,'target_log_return':.015*np.sin(j/11)})
    dev=stream[:280];joint=calibrate(stream,initial,params,'residual_ewma','variable_gamma_aci')
    first=calibrate(dev,initial,params,'residual_ewma','variable_gamma_aci')
    assert len(joint)==48 and len(first)==28
    for a,b in zip(joint[:28],first):
        for k in ('lower','upper','alpha_t','return_volatility','gamma_driver'):
            assert a[k]==b[k]
    assert joint[28]['origin_index']==stream[280]['origin_index']

def test_metric_arithmetic_gate():
    m={'n':731,'coverage':.8,'lower_miss_rate':.1,'upper_miss_rate':.1,
       'width_mean':.03,'lower_penalty_mean':.004,'upper_penalty_mean':.005,'winkler_mean':.039}
    sanity_metrics(m)
    m['winkler_mean']+=.01
    with pytest.raises(ValueError,match='ARITHMETIC'):sanity_metrics(m)
