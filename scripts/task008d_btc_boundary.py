"""TASK-008D BTC Development->Validation boundary audit; CPU only, no inference.

Audits immutable BTC input and Dev caches; produces write-once boundary state
fingerprints and transition *probes*. Probes do NOT constitute TSFM predictions.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, os, subprocess, sys
from collections import deque
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from cita2027.btc.development import (DATA_SHA, DEV_SHA, VAL_SHA, MODELS, load_inputs,
    verify_manifest, cross_model, cache_dir, stored_json, DEV_COUNT)
from cita2027.btc.ingest import forecast_context
from cita2027.calibration.online import (Parameters, SCALES, CONTROLLERS,
    signed_score, conformal_order_statistic, interval, calibrate)
from cita2027.utils.provenance import sha256_file, canonical_json_bytes
from cita2027.forecasting.cache import store_or_verify

REPORT_SCHEMA='btc-validation-boundary-lock-v1'
VAL_COUNT=731
EXPECTED_DEV_REPORT='7782834dfa3aaa0e863a82a02c1e49604a493b2be2223467320fb782e0933210'
EXPECTED_COMPARISON='47f697cd1ea193d2fc0c78ba53e69158183cb15d612e44f5c89412bcc113060f'
PROBE_COUNT=3

def digest(obj): return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()

def boundary_plan_check(dev, val, bounded_df):
    if (val['count']!=VAL_COUNT or len(val['origins'])!=VAL_COUNT or val['target_split']!='validation'
        or val['dataset_sha256']!=DATA_SHA or val['calibration_state']!='btc_independent_causal_start_dev_then_validation_carry_forward_no_reset'):
        raise ValueError('BTC_VAL_PLAN_DRIFT')
    for k,v in {'context_length':256,'horizon':1,'samples':100,'quantile_levels':[.1,.5,.9],
                'alpha':.2,'W':252,'gamma0':.01,'ewma_decay':.94,'gamma_driver_bounds':[.5,2.],
                'correction':'expansion_only'}.items():
        if val.get(k)!=v or dev.get(k)!=v: raise ValueError('BTC_PROTOCOL_DRIFT:'+k)
    devlast=dev['origins'][-1]; first=val['origins'][0]
    if (devlast['origin_index']!=1961 or devlast['target_index']!=1962
        or devlast['target_date']!='2022-12-31' or first['origin_index']!=1962
        or first['target_index']!=1963 or first['origin_date']!='2022-12-31'
        or first['target_date']!='2023-01-01' or first['context_first_index']!=1707
        or first['context_last_index']!=1962 or first['seed']!=3989):
        raise ValueError('BTC_BOUNDARY_ORIGIN_DRIFT')
    dates=[date.fromisoformat(x['target_date']) for x in val['origins']]
    expected=[date(2023,1,1)+timedelta(days=i) for i in range(VAL_COUNT)]
    if dates!=expected or dates[-1]!=date(2024,12,31):
        raise ValueError('BTC_WEEKEND_CALENDAR_GAP')
    for idx,p in enumerate(val['origins']):
        if p['target_index']!=1963+idx or p['origin_index']!=1962+idx or p['seed']!=3989+idx:
            raise ValueError('BTC_ABSOLUTE_INDEX_OR_SEED_DRIFT')
        if p['context_last_index']!=p['origin_index'] or p['context_first_index']!=p['origin_index']-255:
            raise ValueError('BTC_CONTEXT_BOUNDARY_DRIFT')
    if bounded_df.iloc[-1]['date']!='2022-12-31' or len(bounded_df)!=1963:
        raise ValueError('BTC_DEV_LAST_DATE_DRIFT')
    ctx=bounded_df.iloc[1707:1963]['log_return'].to_numpy(dtype=np.float64,copy=True)
    if len(ctx)!=256 or not np.isfinite(ctx).all():raise ValueError('BTC_BRIDGE_CONTEXT_NONFINITE')
    if hashlib.sha256(np.asarray(ctx,dtype='<f8').tobytes()).hexdigest()!=first['context_sha256']:
        raise ValueError('BTC_VAL_FIRST_CONTEXT_SHA_DRIFT')
    if date(2022,12,31).weekday()!=5 or date(2023,1,1).weekday()!=6:
        raise ValueError('BTC_WEEKEND_BRIDGE_DRIFT')
    return first

def calendar_static_check(root:Path):
    path=root/'src'/'cita2027'/'forecasting'/'adapters.py'
    content=path.read_text(encoding='utf-8')
    tree=ast.parse(content)
    cls=next((x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Moirai10Adapter'),None)
    if cls is None: raise ValueError('MOIRAI_ADAPTER_MISSING')
    init=next((x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__init__'),None)
    pred=next((x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='predict'),None)
    if init is None or pred is None:raise ValueError('MOIRAI_ADAPTER_METHODS_MISSING')
    init_text=ast.get_source_segment(content,init) or ''
    pred_text=ast.get_source_segment(content,pred) or ''
    # Guard existing experimental contract: synthetic weekdays, no covariates,
    # scalar-context only; do not silently change BDay to daily post hoc.
    for token in ('feat_dynamic_real_dim=0','past_feat_dynamic_real_dim=0','prediction_length=1','context_length=256'):
        if token not in init_text:raise ValueError('MOIRAI_LOCKED_CONFIG_DRIFT:'+token)
    for token in ('pd.Period("2000-01-03", freq="B")','freq="B"','context.astype(np.float32)',
                  'samples[:, 0]'):
        if token not in pred_text:raise ValueError('MOIRAI_SYNTHETIC_INDEX_CONTRACT_DRIFT:'+token)
    if any(t in pred_text for t in ('freq="D"','feat_dynamic_real','datetime.now','pd.Timestamp(')):
        raise ValueError('MOIRAI_DATE_FEATURES_OR_FREQ_CHANGED')
    return {'synthetic_index':'2000-01-03 BDay','date_features_declared':0,
        'forecast_temporal_mapping':'external_absolute_btc_origin_plan',
        'bdays_are_not_actual_btc_target_dates':True,
        'internal_gluonts_transform_inspection':'NOT_PROVEN_BY_STATIC_CHECK',
        'accepted_dev_artifact_impact':'NO_CODE_CHANGE; if adapter frequency changes, invalidate/rebuild affected BTC caches'}

def initial_state(first_context, params):
    first=np.asarray(first_context,dtype=float)
    if first.shape!=(256,) or not np.isfinite(first).all():raise ValueError('BAD_INITIAL_CONTEXT')
    base=max(float(np.mean(first**2)),params.min_scale**2)
    return {'var_ret':base,'var_res':base,'reference':max(np.sqrt(base),params.min_scale),
            'scores':deque(maxlen=params.window),'alpha':float(params.alpha),'processed':0}

def one_transition(state,rec,params,scale,control,*,expect_output):
    # Prediction uses prior state, update sees target only afterward.
    ret=max(np.sqrt(state['var_ret']),params.min_scale)
    res=max(np.sqrt(state['var_res']),params.min_scale)
    s={'none':1.,'residual_ewma':res,'return_ewma':ret}[scale]
    driver=float(np.clip(ret/state['reference'],params.driver_min,params.driver_max))
    gamma=params.gamma*(driver if control=='variable_gamma_aci' else 1.)
    q10,q50,q90,y=(float(rec[k]) for k in ('q10','q50','q90','target_log_return'))
    active=params.alpha if control=='rolling_cqr' else state['alpha']
    observed=None
    if expect_output:
        if len(state['scores'])!=params.window:raise ValueError('INCOMPLETE_ROLLING_BUFFER')
        threshold=conformal_order_statistic(state['scores'],active)
        lo,hi=interval(q10,q90,threshold,s,params.correction)
        miss=int(y<lo or y>hi)
        observed={'lower':lo,'upper':hi,'alpha_t':active,'score_threshold':threshold,
                  'scale':s,'gamma_driver':driver,'gamma_t':0. if control=='rolling_cqr' else gamma,
                  'return_volatility':ret,'residual_volatility':res,'miss':miss}
        if control!='rolling_cqr':
            floor=1./(params.window+1)
            state['alpha']=float(np.clip(state['alpha']+gamma*(params.alpha-miss),floor,1-floor))
    state['scores'].append(signed_score(y,q10,q90)/s)
    state['var_ret']=params.decay*state['var_ret']+(1-params.decay)*y*y
    state['var_res']=params.decay*state['var_res']+(1-params.decay)*(y-q50)**2
    state['processed']+=1
    return observed

def state_payload(state):
    return {'var_ret':state['var_ret'],'var_res':state['var_res'],'reference':state['reference'],
            'alpha_next':state['alpha'],'scores':list(state['scores']),
            'processed':state['processed']}

def restore_state(x,params):
    if x['processed']!=DEV_COUNT or len(x['scores'])!=params.window:
        raise ValueError('INVALID_TRANSFER_STATE')
    return {'var_ret':x['var_ret'],'var_res':x['var_res'],'reference':x['reference'],
            'alpha':x['alpha_next'],'scores':deque(x['scores'],maxlen=params.window),
            'processed':x['processed']}

def regime_boundary(rows,first_context,params):
    # independent scalar replay of the return-EWMA driver and prior 252 sigmas.
    start=initial_state(first_context,params)
    history=deque(maxlen=252)
    for rec in rows:
        sigma=max(np.sqrt(start['var_ret']),params.min_scale)
        history.append(sigma)
        y=float(rec['target_log_return'])
        start['var_ret']=params.decay*start['var_ret']+(1-params.decay)*y*y
    sigma=max(np.sqrt(start['var_ret']),params.min_scale)
    threshold=float(np.quantile(np.asarray(history,dtype=float),.75,method='linear'))
    h=list(history)
    return {'sigma_at_first_validation_origin':sigma,
            'prior_252_origin_sigma_q75':threshold,'is_high':bool(sigma>threshold),
            'prior_sigma_count':len(h),'prior_252_origin_sigmas':h,
            'prior_sigma_history_sha256':digest(h)}

def state_replay(rows,first_context,params):
    results={}
    # Compare independent local state equations against accepted calibrate() on
    # complete DEVELOPMENT stream. No Validation forecasts are invented.
    for scale in SCALES:
        for control in CONTROLLERS:
            key=f'{scale}__{control}'
            s=initial_state(first_context,params)
            for j,rec in enumerate(rows):
                out=one_transition(s,rec,params,scale,control,expect_output=j>=params.window)
                if j==params.window and out is None:raise ValueError('BURNIN_BOUNDARY_OUTPUT_MISSING')
            accepted=calibrate(rows,first_context,params,scale,control)
            if len(accepted)!=DEV_COUNT-params.window:raise ValueError('ACCEPTED_DEV_SCORE_COUNT_DRIFT')
            last=accepted[-1]
            # A step-by-step exact independent replay of all 1209 evaluated Dev rows
            # is tested separately in focused unit tests. For this audit, verify
            # output on last dev target by re-running an independent prefix.
            snap=initial_state(first_context,params)
            for j,r in enumerate(rows[:-1]):one_transition(snap,r,params,scale,control,expect_output=j>=params.window)
            observed=one_transition(snap,rows[-1],params,scale,control,expect_output=True)
            for field in ('lower','upper','alpha_t','scale','gamma_driver','gamma_t','return_volatility','residual_volatility','miss'):
                if not np.isclose(float(observed[field]),float(last[field]),rtol=1e-12,atol=1e-12):
                    raise ValueError('INDEPENDENT_REPLAY_DIFFERS_FROM_ACCEPTED_CALIBRATION:'+key+':'+field)
            payload=state_payload(s)
            # Independence from any future: first next prediction depends only on Dev.
            results[key]={'state_sha256':digest(payload), 'state':payload,
                'next_driver':float(np.clip(max(np.sqrt(s['var_ret']),params.min_scale)/s['reference'],params.driver_min,params.driver_max))}
    return results

def synthetic_probe_inputs(rows, val_plan, count=PROBE_COUNT):
    # Use *existing Development quantiles* as artificial forecast probes;
    # targets are also Development targets (never Validation target observations).
    # This verifies split/rejoin state transfer without looking ahead at BTC Validation.
    tail=rows[-count:]
    return [{**dict(r),'origin_index':val_plan['origins'][j]['origin_index'],
             'target_index':val_plan['origins'][j]['target_index'],
             'origin_date':val_plan['origins'][j]['origin_date'],
             'target_date':val_plan['origins'][j]['target_date'],
             'seed':val_plan['origins'][j]['seed']} for j,r in enumerate(tail)]

def probe_transition(recs,first_context,states,plan,params):
    sample=synthetic_probe_inputs(recs,plan)
    outputs={}
    for key,item in states.items():
        scale,controller=key.split('__')
        local=restore_state(item['state'],params)
        steps=[]
        for rec in sample:
            out=one_transition(local,rec,params,scale,controller,expect_output=True)
            steps.append({'origin_date':rec['origin_date'],'target_date':rec['target_date'],
                          'state_before_sha256':digest(state_payload(restore_state(item['state'],params))) if not steps else steps[-1]['state_after_sha256'],
                          'state_after_sha256':digest(state_payload(local)),
                          'alpha_t':out['alpha_t'],'gamma_driver':out['gamma_driver']})
        # Compare genuine carry-forward versus fresh reset. Reset must differ.
        reset=initial_state(first_context,params)
        if digest(state_payload(reset))==digest(item['state']):raise ValueError('VALIDATION_STATE_WOULD_RESET')
        outputs[key]={'three_step_probe_terminal_sha256':digest(state_payload(local)),
                      'steps':steps,'probe_source':'DEVELOPMENT_LAST_3_TARGETS_AND_QUANTILES_NOT_VALIDATION_FORECASTS'}
    return outputs

def audit(root:Path):
    root=root.resolve()
    df,dev,ident,_,_=load_inputs(root)
    plan=stored_json(root/'artifacts'/'btcusdt_data'/'plans'/f'{VAL_SHA}.json',VAL_SHA)
    first=boundary_plan_check(dev,plan,df)
    adapter=calendar_static_check(root)
    # Strong provenance check: all accepted Dev model cache records and manifests.
    rr={m:verify_manifest(root,m,dev,ident[m],df)[1] for m in MODELS}
    cross_model(rr['chronos'],rr['moirai'])
    params=Parameters()
    if asdict(params)!={'alpha':.20,'window':252,'decay':.94,'gamma':.01,
                        'min_scale':1e-6,'driver_min':.5,'driver_max':2.,'correction':'expansion_only'}:
        raise ValueError('ACCEPTED_CALIBRATION_DEFAULT_DRIFT')
    report_dir=root/'artifacts'/'calibration'/'btcusdt'/'development'/DEV_SHA
    report_file=report_dir/'btc_development_calibration_report.json'
    cmp_file=report_dir/'task008b_spy_btc_contrasts.json'
    if sha256_file(report_file)!=EXPECTED_DEV_REPORT or sha256_file(cmp_file)!=EXPECTED_COMPARISON:
        raise ValueError('BTC_ACCEPTED_DEVELOPMENT_REPORT_SHA_DRIFT')
    if (json.loads(report_file.read_text(encoding='utf-8'))['development_plan_sha256']!=DEV_SHA):
        raise ValueError('BTC_ACCEPTED_REPORT_PLAN_DRIFT')
    base=forecast_context(df,501)
    references={m:state_replay(rr[m],base,params) for m in MODELS}
    regime=regime_boundary(rr['chronos'],base,params)
    # The volatility regime depends only on the observed return stream, hence same across models.
    if regime!=regime_boundary(rr['moirai'],base,params):raise ValueError('CROSS_MODEL_RETURN_REGIME_DRIFT')
    probes={m:probe_transition(rr[m],base,references[m],plan,params) for m in MODELS}
    result={'schema_version':REPORT_SCHEMA,'status':'BTC_BOUNDARY_STATE_AUDIT_PASS_LIMITED_CALENDAR_PROVENANCE',
        'dataset_sha256':DATA_SHA,'development_plan_sha256':DEV_SHA,'validation_plan_sha256':VAL_SHA,
        'development_forecasts_per_model':DEV_COUNT,'validation_target_count':VAL_COUNT,
        'first_validation_origin':first['origin_date'],'first_validation_target':first['target_date'],
        'context_first_index':first['context_first_index'],'context_last_index':first['context_last_index'],
        'first_validation_context_sha256':first['context_sha256'],'seed_first':first['seed'],
        'model_calendar':adapter,'regime_boundary':regime,
        'state_transfer':references,'transitions_protocol_probes':probes,
        'validation_forecasts_generated':False,'validation_target_values_read':False,
        'limitations':['Moirai internal GluonTS transformation/date feature behavior not proven by static AST audit',
                       'Synthetic state probes reuse Development target/quantiles, NOT actual Validation forecasts',
                       'Temporal mapping to UTC daily remains external to Moirai synthetic BDay index'],
        'btc_historical_point_in_time_verified':False,'contamination_free_verified':False,
        'scientific_clearance':False,'final_test_used':False,
        'go_btc_validation':'GO_TECHNICAL_WITH_SYNTHETIC_BDAY_LIMITATION_NO_METHOD_CHANGE'}
    payload=canonical_json_bytes(result)
    path=root/'artifacts'/'btcusdt_boundary'/'task008d'/f'{hashlib.sha256(payload).hexdigest()}.json'
    store_or_verify(path,result)
    return {'status':result['status'],'boundary_report':str(path),'report_sha256':sha256_file(path),
        'validation_plan_sha256':VAL_SHA,'dev_forecasts_verified':{m:len(rr[m]) for m in MODELS},
        'state_cells_per_model':len(SCALES)*len(CONTROLLERS),'validation_forecasts_generated':False,
        'go_btc_validation':result['go_btc_validation'], 'scientific_clearance':False}

def calendar_runtime_probe():
    """Moirai-vENV-only: instantiate real GluonTS ListDataset, no checkpoint/GPU."""
    from gluonts.dataset.common import ListDataset
    import pandas as pd
    ds=ListDataset([{'start':pd.Period('2000-01-03',freq='B'),
                     'target':np.linspace(-.01,.01,256,dtype=np.float32)}],freq='B')
    entries=list(ds)
    if len(entries)!=1 or list(entries[0].keys())!=['start','target']:
        raise ValueError('MOIRAI_GLUONTS_DATASET_FEATURE_DRIFT')
    x=entries[0]
    if str(x['start'])!='2000-01-03' or len(x['target'])!=256:
        raise ValueError('MOIRAI_GLUONTS_SYNTHETIC_START_DRIFT')
    next_start=x['start']+256
    if str(next_start)!=(pd.Period('2000-01-03',freq='B')+256).strftime('%Y-%m-%d'):
        raise ValueError('MOIRAI_GLUONTS_START_ADVANCE_DRIFT')
    return {'status':'MOIRAI_GLUONTS_CPU_INPUT_PROBE_PASS',
        'input_feature_keys':list(x.keys()), 'synthetic_start':str(x['start']),
        'synthetic_next_start':str(next_start),'context_length':256,
        'real_utc_target_mapping':'external_locked_origin_plan_not_synthetic_date',
        'no_gpu_inference':True}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--_moirai-calendar-probe',action='store_true')
    a=p.parse_args()
    if a._moirai_calendar_probe:
        print(json.dumps(calendar_runtime_probe(),sort_keys=True)); return
    root=a.root.resolve()
    env={**os.environ,'PYTHONPATH':os.pathsep.join([str(root/'src'),str(root/'scripts')])}
    tests=root/'tests'/'test_task008d_btc_boundary.py'
    subprocess.run([sys.executable,'-m','pytest','-q',str(tests)],cwd=root,env=env,check=True)
    moirai_py=root/'.venv-moirai'/'Scripts'/'python.exe'
    if not moirai_py.is_file():raise ValueError('MISSING_MOIRAI_ENV_FOR_CPU_CALENDAR_PROBE')
    subprocess.run([str(moirai_py),str(root/'scripts'/'task008d_btc_boundary.py'),
         '--_moirai-calendar-probe'],cwd=root,env=env,check=True)
    print(json.dumps(audit(root),indent=2,sort_keys=True))
if __name__=='__main__':main()
