"""TASK-008B: BTC development only. No SPY modifications, Validation or Final Test I/O.

Scientific contract: frozen local checkpoints, strictly causal origin contexts, complete
per-origin immutable cache, CPU-only W=252 calibration, exploratory paired statistics.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import yaml

from cita2027.btc.ingest import forecast_context, origin_plan
from cita2027.calibration.online import Parameters, SCALES, CONTROLLERS, calibrate, interval_metrics
from cita2027.calibration.development_analysis import (
    score_arrays, native_rows, signed_rolling_diagnostic, circular_block_indices,
    paired_ci, validated_pair_keys,
)
from cita2027.calibration.diagnostic_correction import causal_regime_indices, masked_bootstrap_ci
from cita2027.forecasting.adapters import seed_inference
from cita2027.forecasting.cache import store_or_verify, write_once_atomic
from cita2027.forecasting.protocol import check_quantiles
from cita2027.utils.provenance import sha256_file, canonical_json_bytes

DATA_SHA = '05088f11c7929995ef6dd28e5135669ebaf37465928c62c826a15f245520299a'
SOURCE_MANIFEST_SHA = '76bbda3bef3be823324f237b5f95354ea55b0e75085a5d9d52b64351fe5f0f5f'
SOURCE_QUALITY_SHA = 'f5887a4d0554dd89a3338af12caa09e35f009a806f3ee097636e6caba22e790a'
DEV_SHA = 'c8737fe917c830e6fb853343e684221d9f0c5543cd84807b92b79e7ee75aaf90'
VAL_SHA = '2a023e69a46aa952ca0015c2d8a87574616366a016c5cad333204418bda12e84'
DEV_COUNT = 1461
EVAL_COUNT = 1209
GATE_COUNT = 8
MODELS = ('chronos','moirai')
BLOCKS = (10,20,40)
REPS = 1000
SEED = 2027
SCHEMA = 'btc-development-forecast-v1'
SPY_REPORT_SHA = '972a196f7beae4e31a343a1163beb678f76ee29e63362bb9b10171d3ea1ff149'


def stored_json(path: Path, expected_sha: str) -> dict:
    if sha256_file(path) != expected_sha:
        raise ValueError(f'ACCEPTED_SHA_DRIFT: {path}')
    obj = json.loads(path.read_text(encoding='utf-8'))
    if path.read_bytes() != canonical_json_bytes(obj):
        raise ValueError(f'NONCANONICAL_ACCEPTED_JSON: {path}')
    return obj


def load_inputs(root: Path):
    """Read exactly bounded BTC snapshot/plans and frozen checkpoint identity metadata."""
    from plan_full_development import (EXPECTED_SMOKE_CONFIG_SHA, EXPECTED_CHECKPOINT_AUDIT_SHA,
                                       EXPECTED_MODELS)
    from pilot_forecast_cache import identities
    root = root.resolve()
    folder = root / 'artifacts' / 'btcusdt_data'
    csv_path = folder / 'snapshots' / f'{DATA_SHA}.csv'
    if sha256_file(csv_path) != DATA_SHA:
        raise ValueError('LOCKED_BTC_DATASET_HASH_MISMATCH')
    src = stored_json(folder/'manifests'/f'{SOURCE_MANIFEST_SHA}.json', SOURCE_MANIFEST_SHA)
    quality = stored_json(folder/'quality'/f'{SOURCE_QUALITY_SHA}.json', SOURCE_QUALITY_SHA)
    dev = stored_json(folder/'plans'/f'{DEV_SHA}.json', DEV_SHA)
    val_path = folder/'plans'/f'{VAL_SHA}.json'
    # Only its hash and fixed metadata: never read any Validation targets/values here.
    if sha256_file(val_path) != VAL_SHA:
        raise ValueError('VALIDATION_PLAN_IDENTITY_DRIFT')
    if (src['processed_sha256'] != DATA_SHA or src['development_plan_sha256'] != DEV_SHA
            or src['validation_plan_sha256'] != VAL_SHA or src['quality_sha256'] != SOURCE_QUALITY_SHA):
        raise ValueError('BTC_SOURCE_PROVENANCE_DRIFT')
    if (quality['quality']['missing_days'] != 0 or quality['quality']['duplicate_days'] != 0
            or quality['quality']['daily_utc_days'] != 2694):
        raise ValueError('BTC_ACCEPTED_QUALITY_DRIFT')
    if dev['count'] != DEV_COUNT or len(dev['origins']) != DEV_COUNT or dev['target_split'] != 'development':
        raise ValueError('BTC_DEVELOPMENT_ORIGIN_PLAN_DRIFT')
    if (dev['dataset_sha256'] != DATA_SHA or dev['seed_policy'] != '2027_plus_origin_index'
            or (dev['context_length'],dev['horizon'],dev['samples'],dev['quantile_levels']) !=
            (256,1,100,[0.1,0.5,0.9])):
        raise ValueError('BTC_FORECAST_PROTOCOL_DRIFT')
    if ((dev['alpha'], dev['W'], dev['gamma0'], dev['ewma_decay'], dev['correction']) !=
            (0.2,252,0.01,0.94,'expansion_only')):
        raise ValueError('BTC_CALIBRATION_PROTOCOL_DRIFT')
    # Read only warmup + Development rows, never any BTC Validation price or return.
    raw = pd.read_csv(csv_path, float_precision='round_trip', nrows=1963)
    if len(raw)!=1963 or raw.iloc[0]['date']!='2017-08-17' or raw.iloc[-1]['date']!='2022-12-31':
        raise ValueError('BTC_READ_BOUNDARY_DRIFT')
    if raw['date'].duplicated().any() or not raw['date'].is_monotonic_increasing:
        raise ValueError('BTC_ORDERING_DRIFT')
    # Plan is verified against its causal 256-return context WITHOUT reading Validation rows.
    # origin_plan() calculates target split locally using only bounded dataframe.
    rebuilt = origin_plan(raw, DATA_SHA, src['snapshot_id'], 'development')
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(dev):
        raise ValueError('BTC_REBUILT_DEV_PLAN_DIFFERS_FROM_PUBLISHED_PLAN')
    config = root/'configs'/'smoke.yaml'
    audit = root/'artifacts'/'checkpoints'/'hf_audit.json'
    if sha256_file(config)!=EXPECTED_SMOKE_CONFIG_SHA or sha256_file(audit)!=EXPECTED_CHECKPOINT_AUDIT_SHA:
        raise ValueError('ACCEPTED_CHECKPOINT_CONFIG_AUDIT_DRIFT')
    conf = yaml.safe_load(config.read_text(encoding='utf-8'))
    if ((conf['context_length'],conf['horizon'],conf['samples'],conf['seed'],conf['quantiles'])!=
            (256,1,100,2027,[0.1,0.5,0.9])):
        raise ValueError('FROZEN_INFERENCE_PROTOCOL_DRIFT')
    checkpoints = json.loads(audit.read_text(encoding='utf-8'))
    identities_by_model = {}
    for model in MODELS:
        cls, model_id, rev, files = identities(conf, checkpoints, model)
        weight = next(iter(files.values()))
        if (rev,weight)!=EXPECTED_MODELS[model]:
            raise ValueError(f'ACCEPTED_CHECKPOINT_REVISION_DRIFT_{model}')
        identities_by_model[model] = (cls,model_id,rev,weight,files)
    return raw,dev,identities_by_model,conf,checkpoints


def cache_dir(root: Path, model: str) -> Path:
    return root/'artifacts'/'forecast_cache'/'btcusdt'/'development'/'runs'/DEV_SHA/model


def record_check(r:dict, p:dict, model:str, model_id:str, rev:str, weight:str, df:pd.DataFrame):
    expected_keys = {'schema_version','model','plan_sha256','dataset_sha256',
        'source_manifest_sha256','checkpoint_id','checkpoint_commit',
        'checkpoint_weight_sha256','origin_index','target_index','origin_date',
        'target_date','context_first_index','context_last_index','context_sha256',
        'target_log_return','q10','q50','q90','seed','context_length','horizon',
        'samples','quantile_levels','target_units','elapsed_seconds',
        'peak_allocated_mib','replay_checked','replay_max_abs_diff',
        'contamination_free_verified','btc_historical_point_in_time_verified',
        'scientific_clearance'}
    if set(r) != expected_keys:
        raise ValueError('BTC_RECORD_SCHEMA_FIELD_SET_DRIFT')
    if type(r['replay_checked']) is not bool:
        raise ValueError('BTC_REPLAY_BOOLEAN_TYPE_DRIFT')
    i = p['origin_index']; j=p['target_index']
    ctx = forecast_context(df, i)
    if j!=i+1 or i!=int(i) or i<256 or j>=len(df): raise ValueError('INVALID_BTC_INDEX')
    if type(r.get('origin_index')) is not int or type(r.get('seed')) is not int:
        raise ValueError('NUMPY_OR_STRING_INDEX_SEED_FORBIDDEN')
    fields = {'schema_version':SCHEMA, 'model':model, 'plan_sha256':DEV_SHA,
              'dataset_sha256':DATA_SHA,'source_manifest_sha256':SOURCE_MANIFEST_SHA,
              'checkpoint_id':model_id,'checkpoint_commit':rev,'checkpoint_weight_sha256':weight,
              'context_length':256,'horizon':1,'samples':100,'quantile_levels':[.1,.5,.9],
              'target_units':'daily_log_return_decimal',
              'origin_index':i,'target_index':j,'origin_date':p['origin_date'],
              'target_date':p['target_date'],'context_first_index':p['context_first_index'],
              'context_last_index':p['context_last_index'],'context_sha256':p['context_sha256'],
              'seed':2027+i,'contamination_free_verified':False,
              'btc_historical_point_in_time_verified':False,'scientific_clearance':False}
    for k,v in fields.items():
        if r.get(k)!=v: raise ValueError(f'BTC_RECORD_IDENTITY_DRIFT {model}:{i}:{k}')
    if sha256(np.asarray(ctx,dtype='<f8').tobytes()).hexdigest()!=p['context_sha256']:
        raise ValueError('BTC_CONTEXT_RECONSTRUCTION_DRIFT')
    target=float(df.iloc[j]['log_return'])
    if not np.isfinite(target) or float(r.get('target_log_return',float('nan'))).hex()!=target.hex():
        raise ValueError('BTC_REALIZED_TARGET_ALIGNMENT_DRIFT')
    check_quantiles([r['q10'],r['q50'],r['q90']])
    for k in ('elapsed_seconds','peak_allocated_mib','replay_max_abs_diff'):
        if not np.isfinite(r[k]) or r[k]<0: raise ValueError(f'INVALID_BTC_DIAGNOSTIC_{k}')
    if r.get('replay_checked') != (i==p['origin_index'] and i in (DEV_FIRST_ORIGIN,DEV_FIRST_ORIGIN+1)):
        raise ValueError('BTC_DETERMINISTIC_REPLAY_FLAG_DRIFT')
    if r['replay_max_abs_diff']>1e-5 or (not r['replay_checked'] and r['replay_max_abs_diff']!=0):
        raise ValueError('BTC_DETERMINISTIC_REPLAY_TOLERANCE_DRIFT')

# BTC daily calendar starts 2017-08-17 at row_index 0. Development target Jan 1 2019
# is row_index 502, so the bridge forecast origin is absolute index 501.
DEV_FIRST_ORIGIN=501


def records(root, model, plan, identity, df, partial=True):
    _,model_id,rev,weight,_ = identity
    path=cache_dir(root,model)/'records'
    actual={f.name for f in path.glob('*.json')} if path.exists() else set()
    expected={f'{row["origin_index"]:06d}.json' for row in plan['origins']}
    if actual-expected: raise ValueError('UNPLANNED_OR_DUPLICATE_BTC_RECORDS')
    out=[]
    for row in plan['origins']:
        file=path/f'{row["origin_index"]:06d}.json'
        if not file.is_file():
            if not partial: raise ValueError('MISSING_BTC_FORECAST_RECORD')
            break
        rec=stored_json_unpinned(file)
        record_check(rec,row,model,model_id,rev,weight,df)
        out.append(rec)
    if len(out)!=len(actual): raise ValueError('NONCONTIGUOUS_BTC_FORECAST_RECORD_PREFIX')
    return out


def stored_json_unpinned(path:Path)->dict:
    obj=json.loads(path.read_text(encoding='utf-8'))
    if path.read_bytes()!=canonical_json_bytes(obj):
        raise ValueError(f'NONCANONICAL_IMMUTABLE_RECORD: {path}')
    return obj


def cross_model(a:list[dict],b:list[dict]):
    if len(a)!=len(b): raise ValueError('CROSS_MODEL_RECORD_COUNT_DRIFT')
    for x,y in zip(a,b):
        for k in ('origin_index','target_index','origin_date','target_date',
                  'context_sha256','seed','target_log_return'):
            if x[k]!=y[k]:raise ValueError(f'CROSS_MODEL_BTC_ALIGNMENT_DRIFT_{k}')


@contextmanager
def exclusive_lock(path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        fd=os.open(path, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'BTC_WORKER_LOCK_EXISTS_INSPECT_PID_BEFORE_REMOVAL: {path}') from None
    try:
        os.write(fd,f'pid={os.getpid()}\n'.encode())
    finally:
        os.close(fd)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def verify_manifest(root, model, plan, identity, df):
    path=cache_dir(root,model)
    manifest_file=path/'manifest.json'
    if not manifest_file.is_file(): raise ValueError(f'BTC_MODEL_MANIFEST_MISSING:{model}')
    manifest=stored_json_unpinned(manifest_file)
    a=records(root,model,plan,identity,df,partial=False)
    csv_bytes=pd.DataFrame(a).to_csv(index=False,float_format='%.17g').encode('utf-8')
    if (path/'predictions.csv').read_bytes()!=csv_bytes:raise ValueError('BTC_PREDICTIONS_CSV_RECORD_MISMATCH')
    _,model_id,rev,weight,_=identity
    expected={'schema_version':'btc-development-manifest-v1','status':'BTC_DEV_CACHE_COMPLETE_NOT_SCIENTIFIC_CLEARANCE',
              'model':model,'count':DEV_COUNT,'dataset_sha256':DATA_SHA,'plan_sha256':DEV_SHA,
              'source_manifest_sha256':SOURCE_MANIFEST_SHA,'checkpoint_id':model_id,
              'checkpoint_commit':rev,'checkpoint_weight_sha256':weight,
              'record_sha256':{f'{r["origin_index"]:06d}.json':sha256_file(path/'records'/f'{r["origin_index"]:06d}.json') for r in a},
              'predictions_sha256':sha256_file(path/'predictions.csv'),
              'context_length':256,'horizon':1,'samples':100,'quantile_levels':[.1,.5,.9],
              'seed_policy':'2027_plus_absolute_btc_origin_index',
              'contamination_free_verified':False,'btc_historical_point_in_time_verified':False,
              'scientific_clearance':False,'validation_read':False,'final_test_read':False}
    if any(manifest.get(k)!=v for k,v in expected.items()) or set(manifest)!=set(expected):
        raise ValueError('BTC_MODEL_MANIFEST_INTEGRITY_FAIL')
    return manifest,a


def run_worker(root:Path,model:str,limit:int):
    from full_development_cache import preflight_worker
    from pilot_forecast_cache import verify_local_checkpoint
    import gc
    import torch
    df,plan,ident,conf,audit=load_inputs(root)
    if model not in MODELS or limit not in (GATE_COUNT,DEV_COUNT):
        raise ValueError('BTC_WORKER_UNAUTHORIZED_SCOPE')
    cls,model_id,rev,weight,files=ident[model]
    preflight_worker(root,conf,audit,model)
    if verify_local_checkpoint(model_id,rev,files)!=weight:raise ValueError('BTC_LOCAL_CHECKPOINT_SHA_DRIFT')
    run_folder=cache_dir(root,model)
    with exclusive_lock(run_folder/'.btc-dev.lock'):
        old=records(root,model,plan,ident[model],df,partial=True)
        if (run_folder/'manifest.json').exists():
            verify_manifest(root,model,plan,ident[model],df)
            return {'model':model,'status':'ALREADY_COMPLETE_VERIFIED','records':DEV_COUNT}
        if len(old)>limit and limit==GATE_COUNT:
            return {'model':model,'status':'GATE_PREFIX_ALREADY_PRESENT','records':len(old)}
        adapter=None; made=0
        try:
            for n,row in enumerate(plan['origins'][len(old):limit],start=len(old)):
                i=int(row['origin_index']); seed=int(row['seed'])
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
                replay=n<2
                diff=0.0
                if replay:
                    seed_inference(seed)
                    q2=adapter.predict(context)
                    diff=float(np.max(np.abs(q-q2)))
                    if not np.allclose(q,q2,atol=1e-5,rtol=0):
                        raise ValueError(f'BTC_DETERMINISTIC_REPLAY_FAIL:{model}:{i}')
                rec={'schema_version':SCHEMA,'model':model,'plan_sha256':DEV_SHA,
                     'dataset_sha256':DATA_SHA,'source_manifest_sha256':SOURCE_MANIFEST_SHA,
                     'checkpoint_id':model_id,'checkpoint_commit':rev,'checkpoint_weight_sha256':weight,
                     'origin_index':i,'target_index':int(row['target_index']),
                     'origin_date':row['origin_date'],'target_date':row['target_date'],
                     'context_first_index':int(row['context_first_index']),
                     'context_last_index':int(row['context_last_index']),
                     'context_sha256':row['context_sha256'],
                     'target_log_return':float(df.iloc[int(row['target_index'])]['log_return']),
                     'q10':float(q[0]),'q50':float(q[1]),'q90':float(q[2]),
                     'seed':seed,'context_length':256,'horizon':1,'samples':100,
                     'quantile_levels':[.1,.5,.9],'target_units':'daily_log_return_decimal',
                     'elapsed_seconds':elapsed,'peak_allocated_mib':peak,
                     'replay_checked':bool(replay),'replay_max_abs_diff':float(diff),
                     'contamination_free_verified':False,
                     'btc_historical_point_in_time_verified':False,'scientific_clearance':False}
                record_check(rec,row,model,model_id,rev,weight,df)
                write_once_atomic(run_folder/'records'/f'{i:06d}.json',canonical_json_bytes(rec))
                made+=1
                if made%50==0 or (limit==GATE_COUNT and n==GATE_COUNT-1):
                    print(json.dumps({'model':model,'progress':'BTC_FORECAST_RECORDS_WRITTEN',
                                     'new':made,'total_prefix':n+1,'limit':limit}),flush=True)
        finally:
            if adapter is not None:
                del adapter
                gc.collect()
                torch.cuda.empty_cache()
        now=records(root,model,plan,ident[model],df,partial=True)
        if len(now)<limit:raise ValueError('BTC_WORKER_PREFIX_INCOMPLETE')
        if limit==GATE_COUNT:
            return {'model':model,'status':'BTC_SMALL_BATCH_GATE_MODEL_PASS','records':len(now),'new':made}
        if len(now)!=DEV_COUNT:raise ValueError('BTC_FULL_DEV_RECORD_COUNT_INVALID')
        csv=pd.DataFrame(now).to_csv(index=False,float_format='%.17g').encode('utf-8')
        csv_file=run_folder/'predictions.csv'
        if csv_file.exists():
            if csv_file.read_bytes()!=csv:raise ValueError('BTC_IMMUTABLE_PREDICTIONS_DRIFT')
        else:
            write_once_atomic(csv_file,csv)
        manifest={'schema_version':'btc-development-manifest-v1','status':'BTC_DEV_CACHE_COMPLETE_NOT_SCIENTIFIC_CLEARANCE',
                  'model':model,'count':DEV_COUNT,'dataset_sha256':DATA_SHA,'plan_sha256':DEV_SHA,
                  'source_manifest_sha256':SOURCE_MANIFEST_SHA,'checkpoint_id':model_id,
                  'checkpoint_commit':rev,'checkpoint_weight_sha256':weight,
                  'record_sha256':{f'{r["origin_index"]:06d}.json':sha256_file(run_folder/'records'/f'{r["origin_index"]:06d}.json') for r in now},
                  'predictions_sha256':sha256_file(csv_file),
                  'context_length':256,'horizon':1,'samples':100,'quantile_levels':[.1,.5,.9],
                  'seed_policy':'2027_plus_absolute_btc_origin_index',
                  'contamination_free_verified':False,'btc_historical_point_in_time_verified':False,
                  'scientific_clearance':False,'validation_read':False,'final_test_read':False}
        store_or_verify(run_folder/'manifest.json',manifest)
        verify_manifest(root,model,plan,ident[model],df)
        return {'model':model,'status':'BTC_DEV_CACHE_COMPLETE','records':DEV_COUNT,'new':made,
                'manifest_sha256':sha256_file(run_folder/'manifest.json')}


def validate_small_gate(root:Path):
    df,plan,ident,_,_=load_inputs(root)
    out={m:records(root,m,plan,ident[m],df) for m in MODELS}
    if any(len(out[m])<GATE_COUNT for m in MODELS):raise ValueError('SMALL_BATCH_BOTH_MODELS_INCOMPLETE')
    cross_model(out['chronos'][:GATE_COUNT],out['moirai'][:GATE_COUNT])
    for m in MODELS:
        for r in out[m][:GATE_COUNT]:
            if r['origin_index'] < DEV_FIRST_ORIGIN + 2 and not r['replay_checked']:
                raise ValueError('SMALL_BATCH_DETERMINISM_MISSING')
    gate={'schema_version':'btc-small-batch-gate-v1',
          'status':'BTC_SMALL_BATCH_CROSS_MODEL_GATE_PASS',
          'dataset_sha256':DATA_SHA,'development_plan_sha256':DEV_SHA,
          'records_per_model':GATE_COUNT, 'first_absolute_btc_origin_index':DEV_FIRST_ORIGIN,
          'seed_first':SEED+DEV_FIRST_ORIGIN,
          'models':{m:{'record_sha256':{f'{r["origin_index"]:06d}.json':
                       sha256_file(cache_dir(root,m)/'records'/f'{r["origin_index"]:06d}.json')
                       for r in out[m][:GATE_COUNT]}} for m in MODELS},
          'contamination_free_verified':False,
          'btc_historical_point_in_time_verified':False,'scientific_clearance':False,
          'validation_used':False,'final_test_used':False}
    gate_file=root/'artifacts'/'forecast_cache'/'btcusdt'/'development'/'gates'/f'{DEV_SHA}.json'
    store_or_verify(gate_file,gate)
    return {'status':gate['status'],'per_model':GATE_COUNT,
            'seed_first':SEED+DEV_FIRST_ORIGIN,'gate_sha256':sha256_file(gate_file),
            'validation_used':False}


def diagnostic_report(root:Path):
    df,plan,ident,_,_=load_inputs(root)
    rows={}
    provenance={}
    for m in MODELS:
        manifest,rr=verify_manifest(root,m,plan,ident[m],df)
        rows[m]=rr
        provenance[m]={'manifest_sha256':sha256_file(cache_dir(root,m)/'manifest.json'),
                       'predictions_sha256':manifest['predictions_sha256']}
    cross_model(rows['chronos'],rows['moirai'])
    params=Parameters()
    first_context=forecast_context(df,DEV_FIRST_ORIGIN)
    regimes=causal_regime_indices(rows['chronos'],first_context)
    if len(regimes['high'])!=EVAL_COUNT:raise ValueError('BTC_REGIME_LENGTH_DRIFT')
    indices={b:circular_block_indices(EVAL_COUNT,REPS,b,SEED) for b in BLOCKS}
    cells_out={}
    trace=[]
    for m in MODELS:
        native=native_rows(rows[m],params.window)
        native_scores=score_arrays(native,params.alpha)
        expected_keys=validated_pair_keys(native)
        per_cell={}
        per_rows={}
        for scale in SCALES:
            for control in CONTROLLERS:
                name=f'{scale}__{control}'
                calibrated=calibrate(rows[m],first_context,params,scale,control)
                if len(calibrated)!=EVAL_COUNT or validated_pair_keys(calibrated)!=expected_keys:
                    raise ValueError('BTC_CALIBRATION_TEMPORAL_ALIGNMENT_DRIFT')
                per_rows[name]=calibrated
                per_cell[name]=score_arrays(calibrated,params.alpha)
        # Shared return-only driver identical across scales AND models.
        shared=[r['gamma_driver'] for r in per_rows['none__rolling_cqr']]
        for name,cal in per_rows.items():
            if [r['gamma_driver'] for r in cal]!=shared:
                raise ValueError('BTC_GAMMA_DRIVER_DIFFERS_ACROSS_CELLS')
        def regime_summary(cal):
            scores=score_arrays(cal,params.alpha)
            out={}
            for label,mask in [('high',regimes['high']),('other',~regimes['high'])]:
                if not mask.any():
                    out[label]={'n':0,'metrics':None}
                    continue
                sub=[r for r,k in zip(cal,mask) if k]
                out[label]={'n':len(sub),'metrics':interval_metrics(sub,params.alpha),
                    'ci95_block20':{key:masked_bootstrap_ci(values,mask,indices[20])
                                  for key,values in scores.items()}}
            return out
        model_out={'raw_q10_q90':{'metrics':interval_metrics(native,params.alpha),
                                    'regimes':regime_summary(native)},'cells':{}}
        for key,cal in per_rows.items():
            sample=per_cell[key]
            scale=key.split('__')[0]
            base_rolling=per_cell['none__rolling_cqr']['winkler']
            fixed=per_cell[f'{scale}__fixed_gamma_aci']['winkler']
            model_out['cells'][key]={'metrics':interval_metrics(cal,params.alpha),
                    'regimes':regime_summary(cal),
                    'paired':{'vs_unscaled_rolling':{str(b):paired_ci(sample['winkler']-base_rolling,idx)
                                                  for b,idx in indices.items()},
                              'vs_same_scale_fixed':{str(b):paired_ci(sample['winkler']-fixed,idx)
                                                    for b,idx in indices.items()},
                              'vs_raw_q10_q90':{str(b):paired_ci(sample['winkler']-native_scores['winkler'],idx)
                                               for b,idx in indices.items()}},
                    'alpha_clip_counts':{'at_lower':sum(r['alpha_t']<=1/253+1e-12 for r in cal),
                                         'at_upper':sum(r['alpha_t']>=252/253-1e-12 for r in cal)}}
            for r,w in zip(cal,sample['winkler']):
                trace.append({'model':m,'cell':key,'origin_index':r['origin_index'],
                              'target_index':r['target_index'],'origin_date':r['origin_date'],
                              'target_date':r['target_date'],'target_log_return':r['target_log_return'],
                              'lower':r['lower'],'upper':r['upper'],'winkler':float(w),
                              'scale':r['scale'],'alpha_t':r['alpha_t'],'gamma_t':r['gamma_t'],
                              'gamma_driver':r['gamma_driver'],'return_volatility':r['return_volatility'],
                              'residual_volatility':r['residual_volatility']})
        signed=signed_rolling_diagnostic(rows[m],params.window,params.alpha)
        signed.pop('rows')
        model_out['signed_rolling_none']=signed
        cells_out[m]=model_out
    # Model equality of the shared driver is checked after both have been calibrated.
    drv={m:[r for r in trace if r['model']==m and r['cell']=='none__rolling_cqr'] for m in MODELS}
    if [r['gamma_driver'] for r in drv['chronos']]!=[r['gamma_driver'] for r in drv['moirai']]:
        raise ValueError('BTC_CROSS_MODEL_GAMMA_DRIVER_DRIFT')
    trace_bytes=pd.DataFrame.from_records(trace).sort_values(['model','cell','origin_index']).to_csv(
        index=False,float_format='%.17g').encode('utf-8')
    out_dir=root/'artifacts'/'calibration'/'btcusdt'/'development'/DEV_SHA
    trace_file=out_dir/'development_1209_per_cell.csv'
    _write_or_verify(trace_file,trace_bytes)
    report={'task':'TASK-008B','status':'BTC_DEV_CALIBRATION_AUDIT_PASS_NOT_SCIENTIFIC_CLEARANCE',
            'dataset_sha256':DATA_SHA,'development_plan_sha256':DEV_SHA,
            'development_origins_per_model':DEV_COUNT,'development_scored_per_cell':EVAL_COUNT,
            'models':cells_out,'forecast_inputs':provenance,
            'regime_definition':'return_EWMA_sigma_gt_prior_252_sigma_q75_causal',
            'regime_counts':{'high':int(regimes['high'].sum()),'other':int((~regimes['high']).sum())},
            'bootstrap':{'blocks':list(BLOCKS),'replicates':REPS,'seed':SEED,
                         'method':'paired_circular_moving_blocks_percentile_95',
                         'multiple_comparison_adjustment':None,'exploratory':True},
            'trace_sha256':sha256_file(trace_file),
            'contamination_free_verified':False,'btc_historical_point_in_time_verified':False,
            'scientific_clearance':False,'validation_used':False,'final_test_used':False,
            'method_selection_performed':False}
    report_file=out_dir/'btc_development_calibration_report.json'
    store_or_verify(report_file,report)
    # Compare within-market paired effects, not absolute cross-market Winkler values.
    spy_file=root/'artifacts'/'calibration'/'task005b'/ '39c14b72d7bd82b815757147838e45fe3df56bddf4616367139cfd5661bd089f'/ 'full_development_W252_expansion_only.json'
    spy=stored_json(spy_file,SPY_REPORT_SHA)
    comparison={'task':'TASK-008B','scope':'within_dataset_paired_contrasts_not_cross_market_absolute_scores',
        'spy_development_report_sha256':SPY_REPORT_SHA,'btc_development_report_sha256':sha256_file(report_file),
        'contrasts':{},'exploratory_no_multiplicity_adjustment':True,
        'contamination_free_verified':False,'btc_historical_point_in_time_verified':False,
        'scientific_clearance':False,'validation_used':False,'final_test_used':False}
    for m in MODELS:
        comparison['contrasts'][m]={}
        for scale in ('residual_ewma','return_ewma'):
            key=f'{scale}__rolling_cqr'
            comparison['contrasts'][m][key]={
                'spy_development_delta_vs_unscaled_rolling':spy['models'][m]['paired_vs_none_rolling'][key],
                'btc_development_delta_vs_unscaled_rolling':cells_out[m]['cells'][key]['paired']['vs_unscaled_rolling']['20'],
                'periods_not_paired':True}
        for scale in SCALES:
            key=f'{scale}__variable_gamma_aci'
            f=spy['models'][m]['cells'][f'{scale}__fixed_gamma_aci']['metrics']['winkler_mean']
            v=spy['models'][m]['cells'][key]['metrics']['winkler_mean']
            comparison['contrasts'][m][key]={
                'spy_development_delta_vs_same_scale_fixed_descriptive':float(v-f),
                'btc_development_delta_vs_same_scale_fixed':cells_out[m]['cells'][key]['paired']['vs_same_scale_fixed']['20'],
                'periods_not_paired':True}
    comparison_file=out_dir/'task008b_spy_btc_contrasts.json'
    store_or_verify(comparison_file,comparison)
    summary={m:{'raw':{k:cells_out[m]['raw_q10_q90']['metrics'][k] for k in ('winkler_mean','coverage')},
                'cells':{key:{'winkler_mean':v['metrics']['winkler_mean'],
                              'coverage':v['metrics']['coverage'],
                              'width_mean':v['metrics']['width_mean']}
                          for key,v in cells_out[m]['cells'].items()}} for m in MODELS}
    return {'status':report['status'],'count_per_model':DEV_COUNT,'evaluated_per_cell':EVAL_COUNT,
            'report_path':str(report_file),'report_sha256':sha256_file(report_file),
            'comparison_sha256':sha256_file(comparison_file),'trace_sha256':sha256_file(trace_file),
            'regime_counts':report['regime_counts'],'summary':summary,
            'contamination_free_verified':False,'btc_historical_point_in_time_verified':False,
            'scientific_clearance':False,'validation_used':False,'final_test_used':False}


def _write_or_verify(path:Path,data:bytes):
    if path.exists():
        if path.read_bytes()!=data:raise ValueError(f'BTC_IMMUTABLE_OUTPUT_DRIFT: {path}')
    else:
        write_once_atomic(path,data)
