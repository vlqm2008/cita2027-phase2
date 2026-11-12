"""Focused TASK-008B tests: BTC causality, state inputs, record provenance, gate sequencing.

Do not rerun accepted SPY or hardware audits; tests use synthetic daily inputs.
"""
from __future__ import annotations
from datetime import date, timedelta
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from cita2027.btc.ingest import forecast_context, origin_plan
from cita2027.btc.development import (
    DATA_SHA, DEV_SHA, DEV_COUNT, DEV_FIRST_ORIGIN, GATE_COUNT,
    record_check, cross_model, records, cache_dir, validate_small_gate,
)
from cita2027.calibration.online import Parameters, calibrate


@pytest.fixture(scope='module')
def fake_df():
    days=pd.date_range('2017-08-17',periods=1963,freq='D')
    closes=100*np.exp(.001*np.arange(1963)+.03*np.sin(np.arange(1963)/11))
    ret=np.r_[np.nan,np.log(closes[1:]/closes[:-1])]
    return pd.DataFrame({'row_index':range(1963),'date':days.strftime('%Y-%m-%d'),
                         'split':['warmup' if i<502 else 'development' for i in range(1963)],
                         'log_return':ret})


@pytest.fixture(scope='module')
def fake_plan(fake_df):
    return origin_plan(fake_df,DATA_SHA,'SYNTHETIC-TEST-SNAPSHOT','development')


def fake_record(df,entry,model='chronos'):
    i=entry['origin_index'];j=entry['target_index']
    return {'schema_version':'btc-development-forecast-v1','model':model,
      'plan_sha256':DEV_SHA,'dataset_sha256':DATA_SHA,
      'source_manifest_sha256':'76bbda3bef3be823324f237b5f95354ea55b0e75085a5d9d52b64351fe5f0f5f',
      'checkpoint_id':'mock','checkpoint_commit':'mockrev','checkpoint_weight_sha256':'mockhash',
      'origin_index':i,'target_index':j,'origin_date':entry['origin_date'],'target_date':entry['target_date'],
      'context_first_index':entry['context_first_index'],'context_last_index':entry['context_last_index'],
      'context_sha256':entry['context_sha256'],'target_log_return':float(df.iloc[j]['log_return']),
      'q10':-.01,'q50':0.,'q90':.01,'seed':2027+i,'context_length':256,'horizon':1,
      'samples':100,'quantile_levels':[.1,.5,.9],'target_units':'daily_log_return_decimal',
      'elapsed_seconds':.1,'peak_allocated_mib':100.,'replay_checked':i in (501,502),
      'replay_max_abs_diff':0.,'contamination_free_verified':False,
      'btc_historical_point_in_time_verified':False,'scientific_clearance':False}


def test_absolute_origin_and_first_bridge(fake_plan):
    assert len(fake_plan['origins'])==DEV_COUNT
    a=fake_plan['origins'][0]
    assert a['origin_index']==501 and a['target_index']==502 and a['seed']==2528
    assert a['origin_date']=='2018-12-31' and a['target_date']=='2019-01-01'
    assert fake_plan['last']['target_date']=='2022-12-31'
    assert fake_plan['development_scored_origins']==1209


def test_context_cannot_see_future_target(fake_df):
    a=forecast_context(fake_df,501)
    changed=fake_df.copy()
    changed.loc[502,'log_return']=1e6
    assert np.array_equal(a,forecast_context(changed,501))
    changed.loc[501,'log_return']*=2
    assert not np.array_equal(a,forecast_context(changed,501))


def test_context_has_256_log_return_decimal_values(fake_df):
    a=forecast_context(fake_df,501)
    assert a.shape==(256,) and np.isfinite(a).all()
    assert a[0]==fake_df.loc[246,'log_return'] and a[-1]==fake_df.loc[501,'log_return']


def test_first_record_rejects_wrong_seed(fake_df,fake_plan):
    entry=fake_plan['origins'][0]
    r=fake_record(fake_df,entry)
    record_check(r,entry,'chronos','mock','mockrev','mockhash',fake_df)
    r['seed']=2027 # SPY-style relative origin bug
    with pytest.raises(ValueError,match='seed'):
        record_check(r,entry,'chronos','mock','mockrev','mockhash',fake_df)


def test_record_rejects_future_target_tamper(fake_df,fake_plan):
    entry=fake_plan['origins'][0]
    r=fake_record(fake_df,entry)
    r['target_log_return']+=.01
    with pytest.raises(ValueError,match='TARGET_ALIGNMENT'):
        record_check(r,entry,'chronos','mock','mockrev','mockhash',fake_df)


def test_record_rejects_numpy_seed_serialization(fake_df,fake_plan):
    entry=fake_plan['origins'][0];r=fake_record(fake_df,entry)
    r['seed']=str(r['seed'])
    with pytest.raises(ValueError,match='NUMPY_OR_STRING'):
        record_check(r,entry,'chronos','mock','mockrev','mockhash',fake_df)


def test_record_rejects_quantile_inversion(fake_df,fake_plan):
    entry=fake_plan['origins'][0];r=fake_record(fake_df,entry)
    r['q10']=2.
    with pytest.raises(ValueError):
        record_check(r,entry,'chronos','mock','mockrev','mockhash',fake_df)


def test_cross_model_alignment_strict(fake_df,fake_plan):
    p=fake_plan['origins'][0]
    a=[fake_record(fake_df,p)];b=[fake_record(fake_df,p,'moirai')]
    cross_model(a,b)
    b[0]['context_sha256']='tampered'
    with pytest.raises(ValueError,match='CROSS_MODEL'):
        cross_model(a,b)


def test_small_gate_fails_without_both_models(fake_df,fake_plan,tmp_path):
    identity=(None,'mock','mockrev','mockhash',{})
    p=fake_plan['origins'][0]
    r=fake_record(fake_df,p)
    from cita2027.utils.provenance import canonical_json_bytes
    from cita2027.forecasting.cache import write_once_atomic
    path=cache_dir(tmp_path,'chronos')/'records'/f'{p["origin_index"]:06d}.json'
    write_once_atomic(path,canonical_json_bytes(r))
    with patch('cita2027.btc.development.load_inputs',return_value=(fake_df,fake_plan,
               {'chronos':identity,'moirai':identity},{},[])):
        with pytest.raises(ValueError,match='SMALL_BATCH_BOTH_MODELS'):
            validate_small_gate(tmp_path)


def test_calibration_has_burnin_and_no_future_feedback(fake_df,fake_plan):
    subset=fake_plan['origins'][:265]
    recs=[fake_record(fake_df,p) for p in subset]
    initial=forecast_context(fake_df,501)
    out=calibrate(recs,initial,Parameters(),'return_ewma','fixed_gamma_aci')
    assert len(out)==13
    assert out[0]['origin_index']==subset[252]['origin_index']
    modified=[dict(r) for r in recs]
    modified[253]['target_log_return']=.9
    alt=calibrate(modified,initial,Parameters(),'return_ewma','fixed_gamma_aci')
    assert (out[1]['lower'],out[1]['upper'])==(alt[1]['lower'],alt[1]['upper'])
    assert out[2]['return_volatility']!=alt[2]['return_volatility']


def test_immutable_record_rejects_mutation(fake_df,fake_plan,tmp_path):
    from cita2027.btc.development import stored_json_unpinned
    from cita2027.forecasting.cache import write_once_atomic
    from cita2027.utils.provenance import canonical_json_bytes
    record=fake_record(fake_df,fake_plan['origins'][0]);p=tmp_path/'record.json'
    write_once_atomic(p,canonical_json_bytes(record))
    assert stored_json_unpinned(p)==record
    p.write_text(json.dumps(record),encoding='utf-8')
    with pytest.raises(ValueError,match='NONCANONICAL'):
        stored_json_unpinned(p)
