"""TASK-008E: locked BTC Validation 2023-2024; no Final Test or SPY writes.

Predict at completed UTC origin, then observe next target only when auditing/updating.
State continuity is achieved by a single Development+Validation replay, with
exact comparison of its entire Development prefix to the immutable Development trace.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from cita2027.btc import development as dev
from cita2027.btc.ingest import forecast_context, origin_plan
from cita2027.calibration.online import Parameters, SCALES, CONTROLLERS, calibrate, interval_metrics
from cita2027.calibration.development_analysis import (
    native_rows, score_arrays, paired_ci, circular_block_indices,
    signed_rolling_diagnostic, validated_pair_keys,
)
from cita2027.calibration.diagnostic_correction import causal_regime_indices, masked_bootstrap_ci
from cita2027.forecasting.adapters import seed_inference
from cita2027.forecasting.protocol import check_quantiles
from cita2027.forecasting.cache import write_once_atomic, store_or_verify
from cita2027.utils.provenance import canonical_json_bytes, sha256_file

DATA_SHA=dev.DATA_SHA
DEV_SHA=dev.DEV_SHA
VAL_SHA=dev.VAL_SHA
MODELS=dev.MODELS
VAL_COUNT=731
FIRST_ORIGIN=1962
BOUNDARY_SHA='48b9b79ff328441312330fb0b907582d3f2635d7d21e739390a3d41c55f10bb1'
DEV_REPORT_SHA='7782834dfa3aaa0e863a82a02c1e49604a493b2be2223467320fb782e0933210'
DEV_COMPARISON_SHA='47f697cd1ea193d2fc0c78ba53e69158183cb15d612e44f5c89412bcc113060f'
SCHEMA='btc-validation-forecast-v1'
BLOCKS=(10,20,40)
REPS=1000
SEED=2027


def val_cache(root:Path,model:str)->Path:
    return root/'artifacts'/'forecast_cache'/'btcusdt'/'validation'/'runs'/VAL_SHA/model


def val_output(root:Path)->Path:
    return root/'artifacts'/'calibration'/'btcusdt'/'validation'/VAL_SHA


def inputs(root:Path):
    """Fail-fast pinned source + boundary; no model/GPU initialization."""
    bounded,development_plan,ident,conf,audit=dev.load_inputs(root)
    report_file=root/'artifacts'/'btcusdt_boundary'/'task008d'/f'{BOUNDARY_SHA}.json'
    boundary=dev.stored_json(report_file,BOUNDARY_SHA)
    if (boundary.get('status')!='BTC_BOUNDARY_STATE_AUDIT_PASS_LIMITED_CALENDAR_PROVENANCE'
        or boundary.get('validation_plan_sha256')!=VAL_SHA
        or boundary.get('validation_forecasts_generated') is not False
        or boundary.get('scientific_clearance') is not False
        or boundary.get('go_btc_validation')!='GO_TECHNICAL_WITH_SYNTHETIC_BDAY_LIMITATION_NO_METHOD_CHANGE'):
        raise ValueError('BOUNDARY_ANCHOR_NOT_ACCEPTED')
    for filename,expected in [('btc_development_calibration_report.json',DEV_REPORT_SHA),
                              ('task008b_spy_btc_contrasts.json',DEV_COMPARISON_SHA)]:
        if sha256_file(root/'artifacts'/'calibration'/'btcusdt'/'development'/DEV_SHA/filename)!=expected:
            raise ValueError('DEVELOPMENT_REPORT_HASH_DRIFT')
    val=dev.stored_json(root/'artifacts'/'btcusdt_data'/'plans'/f'{VAL_SHA}.json',VAL_SHA)
    from task008d_btc_boundary import boundary_plan_check, calendar_static_check
    boundary_plan_check(development_plan,val,bounded)
    # Do not change the accepted synthetic BDay adapter between TASK-008D and inference.
    if calendar_static_check(root)!=boundary['model_calendar']:
        raise ValueError('MOIRAI_ADAPTER_CALENDAR_CONTRACT_DRIFT')
    raw_path=root/'artifacts'/'btcusdt_data'/'snapshots'/f'{DATA_SHA}.csv'
    if sha256_file(raw_path)!=DATA_SHA:raise ValueError('LOCKED_BTC_DATASET_HASH_DRIFT')
    full=pd.read_csv(raw_path,float_precision='round_trip')
    if (len(full)!=2694 or full.iloc[-1]['date']!='2024-12-31'
        or full['date'].duplicated().any() or not full['date'].is_monotonic_increasing
        or full['date'].ge('2025-01-01').any()):
        raise ValueError('VALIDATION_DATASET_SCOPE_OR_CALENDAR_DRIFT')
    if full.iloc[:len(bounded)].to_csv(index=False,float_format='%.17g') != bounded.to_csv(index=False,float_format='%.17g'):
        raise ValueError('DEV_PREFIX_DATA_DRIFT')
    source=dev.stored_json(root/'artifacts'/'btcusdt_data'/'manifests'/f'{dev.SOURCE_MANIFEST_SHA}.json',dev.SOURCE_MANIFEST_SHA)
    rebuilt=origin_plan(full,DATA_SHA,source['snapshot_id'],'validation')
    if canonical_json_bytes(rebuilt)!=canonical_json_bytes(val):
        raise ValueError('REBUILT_VALIDATION_PLAN_DRIFT')
    for j,p in enumerate(val['origins']):
        if (p['origin_index']!=FIRST_ORIGIN+j or p['target_index']!=FIRST_ORIGIN+j+1
            or p['seed']!=SEED+FIRST_ORIGIN+j):raise ValueError('ABSOLUTE_BTC_ORIGIN_DRIFT')
    # Accepted Development manifests & record identities are checked without re-inference.
    dev_records={}
    for model in MODELS:
        _,rr=dev.verify_manifest(root,model,development_plan,ident[model],bounded)
        dev_records[model]=rr
    dev.cross_model(dev_records['chronos'],dev_records['moirai'])
    return full,val,ident,conf,audit,dev_records,boundary


def check_record(rec:dict,p:dict,model:str,identity,df:pd.DataFrame):
    _,model_id,rev,weight,_=identity
    expected={'schema_version':SCHEMA,'model':model,'plan_sha256':VAL_SHA,
              'dataset_sha256':DATA_SHA,'source_manifest_sha256':dev.SOURCE_MANIFEST_SHA,
              'checkpoint_id':model_id,'checkpoint_commit':rev,'checkpoint_weight_sha256':weight,
              'origin_index':p['origin_index'],'target_index':p['target_index'],
              'origin_date':p['origin_date'],'target_date':p['target_date'],
              'context_first_index':p['context_first_index'],'context_last_index':p['context_last_index'],
              'context_sha256':p['context_sha256'],'seed':p['seed'],
              'context_length':256,'horizon':1,'samples':100,'quantile_levels':[.1,.5,.9],
              'target_units':'daily_log_return_decimal','contamination_free_verified':False,
              'btc_historical_point_in_time_verified':False,'scientific_clearance':False}
    allowed=set(expected)|{'q10','q50','q90','target_log_return','elapsed_seconds',
                           'peak_allocated_mib','replay_checked','replay_max_abs_diff'}
    if set(rec)!=allowed or any(rec.get(k)!=v for k,v in expected.items()):
        raise ValueError('VALIDATION_RECORD_METADATA_OR_SCHEMA_DRIFT')
    if type(rec['origin_index']) is not int or type(rec['seed']) is not int or type(rec['replay_checked']) is not bool:
        raise ValueError('VALIDATION_ORIGIN_SEED_FLAG_TYPE_DRIFT')
    i=p['origin_index']; j=p['target_index']
    ctx=forecast_context(df,i)
    if hashlib.sha256(np.asarray(ctx,dtype='<f8').tobytes()).hexdigest()!=p['context_sha256']:
        raise ValueError('VALIDATION_CONTEXT_SHA_MISMATCH')
    y=float(df.iloc[j]['log_return'])
    if float(rec['target_log_return']).hex()!=y.hex() or not np.isfinite(y):
        raise ValueError('VALIDATION_TARGET_MISMATCH')
    check_quantiles([rec['q10'],rec['q50'],rec['q90']])
    for k in ('elapsed_seconds','peak_allocated_mib','replay_max_abs_diff'):
        if not np.isfinite(rec[k]) or rec[k]<0:raise ValueError('INVALID_FORECAST_RUNTIME_DIAGNOSTIC')
    replay=(i in (FIRST_ORIGIN,FIRST_ORIGIN+1))
    if (rec['replay_checked'] is not replay or rec['replay_max_abs_diff']>1e-5
        or (not replay and rec['replay_max_abs_diff']!=0)):
        raise ValueError('VALIDATION_REPLAY_INTEGRITY_FAIL')


def cache_records(root,model,plan,ident,df,*,complete=False):
    path=val_cache(root,model)/'records'
    actual={f.name for f in path.glob('*.json')} if path.exists() else set()
    expected={f'{p["origin_index"]:06d}.json' for p in plan['origins']}
    if actual-expected:raise ValueError('UNPLANNED_OR_DUPLICATE_VALIDATION_RECORDS')
    rows=[]
    for p in plan['origins']:
        f=path/f'{p["origin_index"]:06d}.json'
        if not f.exists():
            if complete:raise ValueError('VALIDATION_CACHE_INCOMPLETE')
            break
        obj=dev.stored_json_unpinned(f)
        check_record(obj,p,model,ident,df)
        rows.append(obj)
    if len(rows)!=len(actual):raise ValueError('NONCONTIGUOUS_VALIDATION_RECORD_PREFIX')
    if complete and len(rows)!=VAL_COUNT:raise ValueError('VALIDATION_CACHE_SIZE_DRIFT')
    return rows


def check_manifest(root,model,plan,ident,df):
    base=val_cache(root,model)
    rows=cache_records(root,model,plan,ident,df,complete=True)
    f=base/'manifest.json'
    if not f.exists():raise ValueError('VALIDATION_MANIFEST_MISSING')
    m=dev.stored_json_unpinned(f)
    _,name,rev,weight,_=ident
    csv=pd.DataFrame(rows).to_csv(index=False,float_format='%.17g').encode('utf-8')
    if (base/'predictions.csv').read_bytes()!=csv:raise ValueError('VALIDATION_PREDICTIONS_DRIFT')
    expected={'schema_version':'btc-validation-manifest-v1',
       'status':'BTC_VALIDATION_CACHE_COMPLETE_NOT_SCIENTIFIC_CLEARANCE',
       'model':model,'count':VAL_COUNT,'dataset_sha256':DATA_SHA,
       'plan_sha256':VAL_SHA,'development_plan_sha256':DEV_SHA,
       'boundary_report_sha256':BOUNDARY_SHA,'source_manifest_sha256':dev.SOURCE_MANIFEST_SHA,
       'checkpoint_id':name,'checkpoint_commit':rev,'checkpoint_weight_sha256':weight,
       'record_sha256':{f'{r["origin_index"]:06d}.json':sha256_file(base/'records'/f'{r["origin_index"]:06d}.json') for r in rows},
       'predictions_sha256':sha256_file(base/'predictions.csv'),
       'context_length':256,'horizon':1,'samples':100,'quantile_levels':[.1,.5,.9],
       'seed_policy':'2027_plus_absolute_btc_origin_index',
       'synthetic_bday_temporal_features_proven_equivalent':False,
       'contamination_free_verified':False,'btc_historical_point_in_time_verified':False,
       'scientific_clearance':False,'final_test_read':False}
    if m!=expected:raise ValueError('VALIDATION_MANIFEST_DRIFT')
    return m,rows


def worker(root,model):
    """Separate .venv process; GPU allocated only after hard preflight."""
    from full_development_cache import preflight_worker
    from pilot_forecast_cache import verify_local_checkpoint
    import torch
    df,plan,ids,conf,audit,_,_=inputs(root)
    if model not in MODELS:raise ValueError('UNKNOWN_MODEL')
    cls,model_id,rev,weight,files=ids[model]
    preflight_worker(root,conf,audit,model)
    if verify_local_checkpoint(model_id,rev,files)!=weight:
        raise ValueError('LOCAL_CHECKPOINT_HASH_DRIFT')
    base=val_cache(root,model)
    with dev.exclusive_lock(base/'.btc-validation.lock'):
        old=cache_records(root,model,plan,ids[model],df)
        if (base/'manifest.json').exists():
            check_manifest(root,model,plan,ids[model],df)
            return {'model':model,'records':VAL_COUNT,'status':'ALREADY_COMPLETE_VERIFIED'}
        adapter=None;made=0
        try:
            for p in plan['origins'][len(old):]:
                i=int(p['origin_index']); seed=int(p['seed'])
                if adapter is None:
                    seed_inference(SEED)
                    adapter=cls(revision=rev,device='cuda',samples=100)
                context=forecast_context(df,i)
                seed_inference(seed)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                start=time.perf_counter()
                q=adapter.predict(context)
                torch.cuda.synchronize()
                elapsed=float(time.perf_counter()-start)
                peak=float(torch.cuda.max_memory_allocated()/2**20)
                check_quantiles(q)
                replay=i in (FIRST_ORIGIN,FIRST_ORIGIN+1)
                diff=0.
                if replay:
                    seed_inference(seed)
                    q2=adapter.predict(context)
                    diff=float(np.max(np.abs(q-q2)))
                    if not np.allclose(q,q2,rtol=0,atol=1e-5):
                        raise ValueError('VALIDATION_DETERMINISTIC_REPLAY_FAIL')
                rec={'schema_version':SCHEMA,'model':model,'plan_sha256':VAL_SHA,
                     'dataset_sha256':DATA_SHA,'source_manifest_sha256':dev.SOURCE_MANIFEST_SHA,
                     'checkpoint_id':model_id,'checkpoint_commit':rev,'checkpoint_weight_sha256':weight,
                     'origin_index':i,'target_index':int(p['target_index']),
                     'origin_date':p['origin_date'],'target_date':p['target_date'],
                     'context_first_index':int(p['context_first_index']),
                     'context_last_index':int(p['context_last_index']),
                     'context_sha256':p['context_sha256'],
                     'target_log_return':float(df.iloc[int(p['target_index'])]['log_return']),
                     'q10':float(q[0]),'q50':float(q[1]),'q90':float(q[2]),
                     'seed':seed,'context_length':256,'horizon':1,'samples':100,
                     'quantile_levels':[.1,.5,.9],'target_units':'daily_log_return_decimal',
                     'elapsed_seconds':elapsed,'peak_allocated_mib':peak,
                     'replay_checked':bool(replay),'replay_max_abs_diff':diff,
                     'contamination_free_verified':False,
                     'btc_historical_point_in_time_verified':False,'scientific_clearance':False}
                check_record(rec,p,model,ids[model],df)
                write_once_atomic(base/'records'/f'{i:06d}.json',canonical_json_bytes(rec))
                made+=1
                if made%50==0:
                    print(json.dumps({'progress':'BTC_VALIDATION_RECORDS_WRITTEN',
                        'model':model,'new':made,'total_prefix':len(old)+made}),flush=True)
        finally:
            if adapter is not None:
                del adapter
                gc.collect();torch.cuda.empty_cache()
        rows=cache_records(root,model,plan,ids[model],df,complete=True)
        csv=pd.DataFrame(rows).to_csv(index=False,float_format='%.17g').encode('utf-8')
        filename=base/'predictions.csv'
        if filename.exists():
            if filename.read_bytes()!=csv:raise ValueError('IMMUTABLE_VALIDATION_PREDICTIONS_DRIFT')
        else:write_once_atomic(filename,csv)
        expected={'schema_version':'btc-validation-manifest-v1',
         'status':'BTC_VALIDATION_CACHE_COMPLETE_NOT_SCIENTIFIC_CLEARANCE',
         'model':model,'count':VAL_COUNT,'dataset_sha256':DATA_SHA,'plan_sha256':VAL_SHA,
         'development_plan_sha256':DEV_SHA,'boundary_report_sha256':BOUNDARY_SHA,
         'source_manifest_sha256':dev.SOURCE_MANIFEST_SHA,'checkpoint_id':model_id,
         'checkpoint_commit':rev,'checkpoint_weight_sha256':weight,
         'record_sha256':{f'{r["origin_index"]:06d}.json':sha256_file(base/'records'/f'{r["origin_index"]:06d}.json') for r in rows},
         'predictions_sha256':sha256_file(filename),'context_length':256,
         'horizon':1,'samples':100,'quantile_levels':[.1,.5,.9],
         'seed_policy':'2027_plus_absolute_btc_origin_index',
         'synthetic_bday_temporal_features_proven_equivalent':False,
         'contamination_free_verified':False,'btc_historical_point_in_time_verified':False,
         'scientific_clearance':False,'final_test_read':False}
        store_or_verify(base/'manifest.json',expected)
        return {'model':model,'records':VAL_COUNT,'new':made,
                'manifest_sha256':sha256_file(base/'manifest.json'),
                'status':'BTC_VALIDATION_CACHE_COMPLETE'}


def sanity_metrics(m):
    if (m['n']!=VAL_COUNT or not np.isclose(m['coverage']+m['lower_miss_rate']+m['upper_miss_rate'],1.,atol=1e-12)
        or not np.isclose(m['width_mean']+m['lower_penalty_mean']+m['upper_penalty_mean'],m['winkler_mean'],atol=1e-12)):
        raise ValueError('VALIDATION_METRIC_ARITHMETIC_FAIL')


def evaluate(root):
    df,plan,ids,_,_,development,boundary=inputs(root)
    validation={};provenance={}
    for model in MODELS:
        manifest,rows=check_manifest(root,model,plan,ids[model],df)
        validation[model]=rows
        provenance[model]={'manifest_sha256':sha256_file(val_cache(root,model)/'manifest.json'),
                           'predictions_sha256':manifest['predictions_sha256']}
    dev.cross_model(validation['chronos'],validation['moirai'])
    for model in MODELS:
        if development[model][-1]['target_index']!=validation[model][0]['origin_index']:
            raise ValueError('DEVELOPMENT_VALIDATION_BOUNDARY_GAP')
    params=Parameters()
    from dataclasses import asdict
    if asdict(params)!={'alpha':.20,'window':252,'decay':.94,'gamma':.01,
                        'min_scale':1e-6,'driver_min':.5,'driver_max':2.,
                        'correction':'expansion_only'}:
        raise ValueError('LOCKED_CALIBRATION_PARAMETERS_DRIFT')
    first_context=forecast_context(df,dev.DEV_FIRST_ORIGIN)
    joint={m:development[m]+validation[m] for m in MODELS}
    regime_full=causal_regime_indices(joint['chronos'],first_context)
    masks=regime_full['high'][-VAL_COUNT:]
    regime_anchor=boundary['regime_boundary']
    for key,observed,expected in (
        ('sigma_at_first_validation_origin',regime_full['sigma'][-VAL_COUNT],regime_anchor['sigma_at_first_validation_origin']),
        ('prior_252_origin_sigma_q75',regime_full['threshold'][-VAL_COUNT],regime_anchor['prior_252_origin_sigma_q75'])):
        if not np.isclose(observed,expected,rtol=1e-12,atol=1e-12):
            raise ValueError('CAUSAL_REGIME_BOUNDARY_STATE_DRIFT:'+key)
    if bool(masks[0]) != regime_anchor['is_high']:
        raise ValueError('CAUSAL_REGIME_BOUNDARY_CLASSIFICATION_DRIFT')
    other_mask=~masks
    if len(masks)!=VAL_COUNT or masks.sum()==0 or other_mask.sum()==0:
        raise ValueError('INVALID_VOLATILITY_REGIME_MASK')
    if not np.array_equal(masks,causal_regime_indices(joint['moirai'],first_context)['high'][-VAL_COUNT:]):
        raise ValueError('CROSS_MODEL_REGIME_MASK_DRIFT')
    index={length:circular_block_indices(VAL_COUNT,REPS,length,SEED) for length in BLOCKS}
    dev_trace_path=root/'artifacts'/'calibration'/'btcusdt'/'development'/DEV_SHA/'development_1209_per_cell.csv'
    dev_report=dev.stored_json(root/'artifacts'/'calibration'/'btcusdt'/'development'/DEV_SHA/'btc_development_calibration_report.json',DEV_REPORT_SHA)
    if sha256_file(dev_trace_path)!=dev_report['trace_sha256']:
        raise ValueError('DEVELOPMENT_ACCEPTED_TRACE_SHA_DRIFT')
    reference=pd.read_csv(dev_trace_path,float_precision='round_trip')
    expected_rows=2*9*dev.EVAL_COUNT
    if len(reference)!=expected_rows:raise ValueError('DEVELOPMENT_REFERENCE_TRACE_COUNT_DRIFT')
    reference=reference.set_index(['model','cell','origin_index']).sort_index()
    for model in MODELS:
        if boundary['state_transfer'][model].keys() != {f'{s}__{c}' for s in SCALES for c in CONTROLLERS}:
            raise ValueError('BOUNDARY_STATE_CELL_SET_DRIFT')
    results={};trace=[]; replication={}
    for model in MODELS:
        per={};arr={}
        native=native_rows(joint[model],params.window)[-VAL_COUNT:]
        native_score=score_arrays(native,params.alpha)
        keys=validated_pair_keys(native)
        for scale in SCALES:
            for controller in CONTROLLERS:
                cell=f'{scale}__{controller}'
                combined=calibrate(joint[model],first_context,params,scale,controller)
                dev_part, val_part=combined[:-VAL_COUNT],combined[-VAL_COUNT:]
                if len(dev_part)!=dev.EVAL_COUNT or len(val_part)!=VAL_COUNT or validated_pair_keys(val_part)!=keys:
                    raise ValueError('VALIDATION_CALIBRATION_RESET_OR_ALIGNMENT')
                # Compare every Development row, not just the terminal snapshot.
                for d in dev_part:
                    ref=reference.loc[(model,cell,d['origin_index'])]
                    for field in ('lower','upper','alpha_t','gamma_t','gamma_driver','scale','return_volatility','residual_volatility'):
                        if not np.isclose(float(d[field]),float(ref[field]),atol=1e-12,rtol=1e-12):
                            raise ValueError(f'CONTINUOUS_REPLAY_DEV_TRACE_DIFFERENCE:{model}:{cell}:{field}')
                # Check independently reconstructed 008D state at the boundary.
                from task008d_btc_boundary import initial_state, one_transition, state_payload, digest
                s=initial_state(first_context,params)
                for j,r in enumerate(development[model]):
                    one_transition(s,r,params,scale,controller,expect_output=j>=params.window)
                if digest(state_payload(s))!=boundary['state_transfer'][model][cell]['state_sha256']:
                    raise ValueError('BOUNDARY_STATE_FINGERPRINT_MISMATCH')
                # At first Validation forecast, controller state before seeing
                # 2023-01-01 must agree with the accepted TASK-008D transfer.
                from task008d_btc_boundary import restore_state
                state=restore_state(boundary['state_transfer'][model][cell]['state'],params)
                predicted=one_transition(state,validation[model][0],params,scale,controller,expect_output=True)
                for field in ('lower','upper','alpha_t','gamma_driver','gamma_t','scale',
                              'return_volatility','residual_volatility'):
                    if not np.isclose(float(predicted[field]),float(val_part[0][field]),rtol=1e-12,atol=1e-12):
                        raise ValueError('BOUNDARY_FIRST_VALIDATION_PREDICTION_STATE_DRIFT:'+cell+':'+field)
                per[cell]=val_part
                arr[cell]=score_arrays(val_part,params.alpha)
        driver=[r['gamma_driver'] for r in per['none__rolling_cqr']]
        if any([r['gamma_driver'] for r in rows]!=driver for rows in per.values()):
            raise ValueError('SHARED_RETURN_GAMMA_DRIVER_DRIFT')
        def regimes(rows):
            scores=score_arrays(rows,params.alpha)
            out={}
            for label,mask in (('high',masks),('other',other_mask)):
                part=[r for r,selected in zip(rows,mask) if selected]
                met=interval_metrics(part,params.alpha)
                out[label]={'n':len(part),'metrics':met,
                            'ci95_block20':{k:masked_bootstrap_ci(v,mask,index[20]) for k,v in scores.items()}}
            return out
        model_out={'raw_q10_q90':{'metrics':interval_metrics(native,params.alpha),'regimes':regimes(native)},'cells':{}}
        sanity_metrics(model_out['raw_q10_q90']['metrics'])
        for cell,rows in per.items():
            scores=arr[cell]
            met=interval_metrics(rows,params.alpha)
            sanity_metrics(met)
            same=arr[f'{cell.split("__")[0]}__fixed_gamma_aci']['winkler']
            unscaled=arr['none__rolling_cqr']['winkler']
            model_out['cells'][cell]={'metrics':met,'regimes':regimes(rows),
                  'paired':{'vs_unscaled_rolling':{str(l):paired_ci(scores['winkler']-unscaled,idx) for l,idx in index.items()},
                            'vs_same_scale_fixed':{str(l):paired_ci(scores['winkler']-same,idx) for l,idx in index.items()},
                            'vs_raw_q10_q90':{str(l):paired_ci(scores['winkler']-native_score['winkler'],idx) for l,idx in index.items()}}}
            for r,w in zip(rows,scores['winkler']):
                trace.append({'model':model,'cell':cell,'origin_index':r['origin_index'],
                              'target_index':r['target_index'],'origin_date':r['origin_date'],
                              'target_date':r['target_date'],'target_log_return':r['target_log_return'],
                              'lower':r['lower'],'upper':r['upper'],'winkler':float(w),
                              'scale':r['scale'],'alpha_t':r['alpha_t'],'gamma_t':r['gamma_t'],
                              'gamma_driver':r['gamma_driver'],'return_volatility':r['return_volatility'],
                              'residual_volatility':r['residual_volatility']})
        # Signed-CQR ablation is scored only on Validation, with all previous
        # signed conformity scores available. Empty sets make Winkler undefined.
        from cita2027.calibration.online import signed_score, conformal_order_statistic
        history=[signed_score(x['target_log_return'],x['q10'],x['q90']) for x in joint[model]]
        signed_rows=[]; empty=[]
        for j in range(len(development[model]),len(joint[model])):
            r=joint[model][j]
            threshold=conformal_order_statistic(history[j-params.window:j],params.alpha)
            lo,hi=r['q10']-threshold,r['q90']+threshold
            if lo>hi:
                empty.append(r['origin_index'])
            else:
                signed_rows.append({'lower':lo,'upper':hi,
                                    'target_log_return':r['target_log_return']})
        signed={'status':'FINITE_SIGNED_INTERVALS' if not empty else 'EMPTY_CONFORMAL_SET_WINKLER_UNDEFINED',
                'empty_count':len(empty),'empty_origin_indices':empty,
                'evaluated':VAL_COUNT,
                'metrics':interval_metrics(signed_rows,params.alpha) if not empty else None,
                'coverage_with_empty_as_miss':None if not empty else
                   sum(x['lower']<=x['target_log_return']<=x['upper'] for x in signed_rows)/VAL_COUNT}
        model_out['signed_rolling_none']=signed
        results[model]=model_out
        replication[model]={}
        for cell,entry in model_out['cells'].items():
            d=dev_report['models'][model]['cells'][cell]['metrics'];v=entry['metrics']
            replication[model][cell]={
                'development_n':d['n'],'validation_n':v['n'],
                'development_winkler':d['winkler_mean'],'validation_winkler':v['winkler_mean'],
                'development_coverage':d['coverage'],'validation_coverage':v['coverage'],
                'validation_width':v['width_mean'],
                'validation_lower_miss':v['lower_miss_rate'],'validation_upper_miss':v['upper_miss_rate'],
                'validation_delta_vs_unscaled_rolling':entry['paired']['vs_unscaled_rolling'],
                'development_delta_vs_unscaled_rolling':dev_report['models'][model]['cells'][cell]['paired']['vs_unscaled_rolling'],
                'validation_delta_vs_same_scale_fixed':entry['paired']['vs_same_scale_fixed'],
                'development_delta_vs_same_scale_fixed':dev_report['models'][model]['cells'][cell]['paired']['vs_same_scale_fixed'],
                'periods_not_paired':True}
    drivers={m:[r['gamma_driver'] for r in trace if r['model']==m and r['cell']=='none__rolling_cqr'] for m in MODELS}
    if drivers['chronos'] != drivers['moirai']:
        raise ValueError('CROSS_MODEL_SHARED_RETURN_GAMMA_DRIVER_DRIFT')
    rawtr=pd.DataFrame(trace).sort_values(['model','cell','origin_index']).to_csv(index=False,float_format='%.17g').encode('utf-8')
    out=val_output(root)
    trace_path=out/'validation_731_per_cell.csv'
    dev._write_or_verify(trace_path,rawtr)
    report={'task':'TASK-008E','status':'BTC_VALIDATION_CALIBRATION_AUDIT_PASS_NOT_SCIENTIFIC_CLEARANCE',
      'validation_plan_sha256':VAL_SHA,'development_plan_sha256':DEV_SHA,'dataset_sha256':DATA_SHA,
      'boundary_report_sha256':BOUNDARY_SHA,'development_calibration_report_sha256':DEV_REPORT_SHA,
      'calibration_state':'causal_continuous_replay_1461_development_then_731_validation_no_reset',
      'validation_forecasts_per_model':VAL_COUNT,'validation_evaluated_per_cell':VAL_COUNT,
      'forecast_inputs':provenance,'regime_counts':{'high':int(masks.sum()),'other':int(other_mask.sum())},
      'regime_rule':'return_EWMA_sigma_gt_prior_252_sigma_q75_causal',
      'bootstrap':{'blocks':list(BLOCKS),'replicates':REPS,'seed':SEED,
                   'method':'paired_circular_moving_blocks_percentile_95',
                   'multiple_comparison_adjustment':None,'exploratory':True},
      'models':results,'trace_sha256':sha256_file(trace_path),
      'synthetic_bday_temporal_features_proven_equivalent':False,
      'model_selection_performed':False,'validation_hyperparameter_tuning':False,
      'btc_historical_point_in_time_verified':False,'contamination_free_verified':False,
      'scientific_clearance':False,'final_test_used':False}
    report_file=out/'btc_validation_calibration_report.json'
    store_or_verify(report_file,report)
    comparison={'task':'TASK-008E','status':'BTC_DEVELOPMENT_VALIDATION_DESCRIPTIVE_CONTRAST_AUDIT_PASS',
      'development_report_sha256':DEV_REPORT_SHA,'validation_report_sha256':sha256_file(report_file),
      'validation_plan_sha256':VAL_SHA,'comparison_scope':'within_period_paired_differences_not_cross_period_paired_test',
      'all_nine_factorial_cells':replication,
      'model_selection_performed':False,'multiple_comparison_adjustment':None,
      'btc_historical_point_in_time_verified':False,'contamination_free_verified':False,
      'scientific_clearance':False,'final_test_used':False}
    comparison_file=out/'task008e_dev_validation_comparison.json'
    store_or_verify(comparison_file,comparison)
    return {'status':report['status'],'validation_count_per_model':VAL_COUNT,
      'total_forecasts':2*VAL_COUNT,'validation_report_sha256':sha256_file(report_file),
      'comparison_sha256':sha256_file(comparison_file),'trace_sha256':sha256_file(trace_path),
      'validation_report_path':str(report_file),'comparison_path':str(comparison_file),
      'regime_counts':report['regime_counts'],'summary':{m:{'raw':{k:results[m]['raw_q10_q90']['metrics'][k] for k in ('winkler_mean','coverage')},
      'cells':{c:{k:v['metrics'][k] for k in ('winkler_mean','coverage','width_mean','lower_miss_rate','upper_miss_rate')}
               for c,v in results[m]['cells'].items()}} for m in MODELS},
      'scientific_clearance':False,'contamination_free_verified':False,
      'btc_historical_point_in_time_verified':False,'final_test_used':False}
