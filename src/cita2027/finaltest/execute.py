"""TASK-011 Phase B, usable ONLY after verified immutable Phase-A protocol lock.

Targets are read after `load_lock()`; at origin i the frozen adapters receive
only [i-255, ..., i]. Every failed forecast halts (no skipped observations).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time
import zipfile

import numpy as np
import pandas as pd

from cita2027.finaltest import lock as L
from cita2027.forecasting.protocol import check_quantiles,load_processed
from cita2027.forecasting.cache import write_once_atomic
from cita2027.utils.provenance import sha256_file
from cita2027.calibration.online import Parameters,calibrate,interval_metrics, SCALES,CONTROLLERS,signed_score,conformal_order_statistic
from cita2027.calibration.development_analysis import score_arrays,paired_ci,circular_block_indices
from cita2027.calibration.diagnostic_correction import causal_regime_indices, masked_bootstrap_ci


def market_dir(root:Path,market:str,lock_sha:str,model:str)->Path:
    return root/'artifacts'/'final_test'/'task011'/'forecast_cache'/lock_sha/market/model


def out_dir(root:Path,lock_sha:str)->Path:
    return root/'artifacts'/'final_test'/'task011'/'evaluation'/lock_sha


def decimal_finite(txt:str,*,positive:bool=False)->float:
    try:x=Decimal(txt)
    except (InvalidOperation,ValueError) as exc:raise ValueError('BTC_BAD_DECIMAL') from exc
    if not x.is_finite() or x<0 or (positive and x<=0):raise ValueError('BTC_NONFINITE_OR_NONPOSITIVE_VALUE')
    f=float(x)
    if not np.isfinite(f):raise ValueError('BTC_NONFINITE_FLOAT')
    return f


def btc_after_lock(root:Path,plan:dict)->pd.DataFrame:
    """2025+ Spot is MICROseconds. Only this Phase B parses OHLC/returns."""
    prev_path=root/'artifacts'/'btcusdt_data'/'snapshots'/f'{L.DATA_SHA}.csv'
    if sha256_file(prev_path)!=L.DATA_SHA:raise ValueError('ACCEPTED_BTC_PREFIX_SHA_DRIFT')
    before=pd.read_csv(prev_path,float_precision='round_trip')
    if len(before)!=2694 or before.iloc[-1]['date']!='2024-12-31':
        raise ValueError('ACCEPTED_BTC_PREFIX_ROW_DRIFT')
    source=plan['source'];new=[]
    expected_dates=[o['target_date'] for o in plan['origins']]
    expected_set=set(expected_dates)
    previous_date='2024-12-31';previous_close=float(before.iloc[-1]['close'])
    for arc in source['archives']:
        z=root/arc['path'];chk=root/arc['checksum_path']
        if sha256_file(z)!=arc['sha256'] or sha256_file(chk)!=arc['checksum_sha256']:
            raise ValueError('FROZEN_BTC_MONTH_SHA_DRIFT')
        name='BTCUSDT-1d-'+arc['month']+'.csv'
        with zipfile.ZipFile(z) as f, f.open(name) as contents:
            text=csv.reader((s.decode('utf-8-sig') for s in contents))
            for r in text:
                if not r:raise ValueError('EMPTY_BTC_KLINE_ROW')
                if r[0].lower() in ('open_time','open time'):continue
                if len(r)!=12 or not re.fullmatch(r'\d{16}',r[0]) or not re.fullmatch(r'\d{16}',r[6]):
                    raise ValueError('BTC_2025_MICROSECONDS_EXPECTED')
                open_us,close_us=int(r[0]),int(r[6]);day_us=86_400_000_000
                if open_us%day_us or close_us!=open_us+day_us-1:
                    raise ValueError('BTC_DAILY_UTC_CANDLE_MISMATCH')
                day=datetime.fromtimestamp(open_us/1_000_000,timezone.utc).date().isoformat()
                # Oct 31 is covered by the Oct monthly archive but excluded by the lock.
                if day not in expected_set:
                    if day=='2026-10-31':continue
                    raise ValueError('UNPLANNED_BTC_TEST_DAY')
                if day!=(pd.Timestamp(previous_date)+pd.Timedelta(days=1)).date().isoformat():
                    raise ValueError('BTC_UTC_DAY_GAP')
                o,h,low,c=(decimal_finite(r[j],positive=True) for j in (1,2,3,4))
                if not (low<=min(o,c)<=max(o,c)<=h):raise ValueError('BAD_BTC_KLINE_OHLC')
                if not re.fullmatch(r'\d+',r[8]):raise ValueError('NONINTEGER_TRADE_COUNT')
                for col in (5,7,9,10):decimal_finite(r[col])
                if not r[11].strip():raise ValueError('BTC_MISSING_IGNORE_FIELD')
                index=len(before)+len(new)
                # Use the previously accepted ratio/log-return convention.
                ret=float(np.log(c/previous_close))
                if not np.isfinite(ret):raise ValueError('BAD_BTC_TEST_RETURN')
                new.append({'row_index':index,'date':day,'split':'final_test',
                   'open_time_ms':open_us//1000,'close_time_ms':close_us//1000,
                   'open':o,'high':h,'low':low,'close':c,'volume':decimal_finite(r[5]),
                   'trades':int(r[8]),'previous_date':previous_date,
                   'previous_row_index':index-1,'log_return':ret})
                previous_date,previous_close=day,c
    if [x['date'] for x in new]!=expected_dates or len(new)!=plan['count']:
        raise ValueError('BTC_LOCKED_TEST_DATE_VALUES_DO_NOT_MATCH')
    df=pd.concat([before,pd.DataFrame(new)],ignore_index=True)
    if len(df)!=2694+plan['count'] or df.iloc[2693]['date']!='2024-12-31':
        raise ValueError('BTC_FINAL_PREFIX_DRIFT')
    return df


def spy_after_lock(root:Path,plan:dict)->pd.DataFrame:
    p=root/plan['source']['source_path']
    if sha256_file(p)!=plan['source']['processed_sha256']:
        raise ValueError('SPY_ACCEPTED_FULL_DATA_SHA_DRIFT')
    df=load_processed(str(p))
    got=[df.iloc[o['target_index']]['date'].date().isoformat() for o in plan['origins']]
    if got!=[o['target_date'] for o in plan['origins']]:
        raise ValueError('SPY_LOCKED_TARGET_DATE_DRIFT')
    for o in plan['origins']:
        i=o['target_index']
        if df.iloc[i]['split']!='final_test' or df.iloc[i-1]['date'].date().isoformat()!=o['origin_date']:
            raise ValueError('SPY_TARGET_SPLIT_OR_ORIGIN_MISMATCH')
        if int(df.iloc[i]['previous_row_index'])!=i-1:
            raise ValueError('SPY_NONCAUSAL_PREVIOUS_INDEX')
        if pd.Timestamp(df.iloc[i]['previous_date'])!=df.iloc[i-1]['date']:
            raise ValueError('SPY_PREVIOUS_SESSION_MISMATCH')
    return df


def load_market_after_lock(root:Path,market:str,lock_sha:str):
    # Executable guard immediately prior to ANY target-containing pandas read.
    _,plans=L.load_lock(root,lock_sha)
    p=plans[market]
    df=spy_after_lock(root,p) if market=='spy' else btc_after_lock(root,p)
    if len(p['origins'])!=p['count']:raise ValueError('TEST_ORIGIN_PLAN_COUNT_DRIFT')
    for j,o in enumerate(p['origins']):
        if o['origin_index']+1!=o['target_index'] or o['seed']!=2027+o['origin_index'] or o['context_last_index']!=o['origin_index'] or o['context_first_index']!=o['origin_index']-255:
            raise ValueError('TEST_CAUSAL_ORIGIN_INDEX_DRIFT')
        if j and o['origin_index']!=p['origins'][j-1]['origin_index']+1:
            raise ValueError('NONCONTIGUOUS_TEST_ORIGINS')
    return df,p


def context_for_origin(df:pd.DataFrame,origin:int)->np.ndarray:
    # No current target or future info. TSFM adapter API ONLY receives this vector.
    x=df.iloc[origin-255:origin+1]['log_return'].to_numpy(dtype=np.float64,copy=True)
    if x.shape!=(256,) or not np.isfinite(x).all():raise ValueError('INVALID_CAUSAL_CONTEXT')
    x.setflags(write=False)
    return x


def context_sha(x:np.ndarray)->str:
    return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()


def target_float(df:pd.DataFrame,idx:int)->float:
    y=float(df.iloc[idx]['log_return'])
    if not np.isfinite(y):raise ValueError('NONFINITE_TEST_TARGET')
    return y


def identity_from_lock(root:Path,model:str,lock_sha:str):
    """Only local checkpoint/metadata; never updates or downloads model weights."""
    lock,_=L.load_lock(root,lock_sha)
    import yaml
    from pilot_forecast_cache import identities
    conf=yaml.safe_load((root/'configs'/'smoke.yaml').read_text('utf-8'))
    audit=json.loads((root/'artifacts'/'checkpoints'/'hf_audit.json').read_text('utf-8'))
    cls,model_id,rev,files=identities(conf,audit,model)
    sha=next(iter(files.values()))
    if lock['checkpoints'][model]!={'id':model_id,'revision':rev,'weight_sha256':sha}:
        raise ValueError('LOCKED_CHECKPOINT_IDENTITY_DRIFT')
    return cls,model_id,rev,sha,files


def validate_record(r:dict,o:dict,df:pd.DataFrame,lock_sha:str,market:str,model:str,identity):
    cls,model_id,revision,weight,_=identity
    i=o['origin_index']
    expected={'schema_version':'task011-final-forecast-v1','lock_sha256':lock_sha,
      'market':market,'model':model,'checkpoint_id':model_id,'checkpoint_commit':revision,
      'checkpoint_weight_sha256':weight,'origin_index':i,'target_index':o['target_index'],
      'origin_date':o['origin_date'],'target_date':o['target_date'],
      'context_first_index':i-255,'context_last_index':i,
      'seed':o['seed'],'context_length':256,'horizon':1,'samples':100,
      'quantile_levels':[.1,.5,.9],'target_units':'daily_log_return_decimal',**L.FLAGS}
    if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('FINAL_RECORD_LOCK_IDENTITY_DRIFT')
    allowed=set(expected)|{'context_sha256','q10','q50','q90','target_log_return',
                             'replay_checked','replay_max_abs_diff','elapsed_seconds','peak_allocated_mib'}
    if set(r)!=allowed:raise ValueError('FINAL_RECORD_SCHEMA_DRIFT')
    if type(r['origin_index']) is not int or type(r['seed']) is not int:
        raise ValueError('FINAL_RECORD_INT_TYPE_DRIFT')
    if r['context_sha256']!=context_sha(context_for_origin(df,i)):
        raise ValueError('FINAL_CONTEXT_SHA_DRIFT')
    if float(r['target_log_return']).hex()!=target_float(df,i+1).hex():
        raise ValueError('FINAL_TARGET_MISMATCH')
    check_quantiles([r['q10'],r['q50'],r['q90']])
    for key in ('replay_max_abs_diff','elapsed_seconds','peak_allocated_mib'):
        if not np.isfinite(r[key]) or r[key]<0:raise ValueError('INVALID_FINAL_RUNTIME')
    if r['replay_max_abs_diff']>1e-5:raise ValueError('REPLAY_GREATER_THAN_TOLERANCE')


def read_records(root:Path,market:str,model:str,lock_sha:str,df,p,identity,*,complete:bool=False):
    base=market_dir(root,market,lock_sha,model)
    path=base/'records'
    expected={f'{o["origin_index"]:06d}.json' for o in p['origins']}
    found={z.name for z in path.glob('*.json')} if path.exists() else set()
    if found-expected:raise ValueError('UNPLANNED_FINAL_FORECAST_FILES')
    rows=[]
    for j,o in enumerate(p['origins']):
        f=path/f'{o["origin_index"]:06d}.json'
        if not f.is_file():
            if complete:raise ValueError('INCOMPLETE_FINAL_RECORDS')
            break
        blob=f.read_bytes();r=json.loads(blob)
        if blob!=L.canonical(r):raise ValueError('NONCANONICAL_FORECAST_RECORD')
        validate_record(r,o,df,lock_sha,market,model,identity)
        if r['replay_checked'] is not (j<2) or (j>=2 and r['replay_max_abs_diff']!=0):
            raise ValueError('FINAL_REPLAY_FLAGS_DRIFT')
        rows.append(r)
    if len(rows)!=len(found):raise ValueError('NONCONTIGUOUS_RESUMED_FORECAST_PREFIX')
    if complete and len(rows)!=p['count']:raise ValueError('FINAL_COUNT_MISMATCH')
    return rows


def check_complete_manifest(root:Path,market,model,lock_sha,df,p,identity):
    base=market_dir(root,market,lock_sha,model)
    rows=read_records(root,market,model,lock_sha,df,p,identity,complete=True)
    source_map={f'{r["origin_index"]:06d}.json':sha256_file(base/'records'/f'{r["origin_index"]:06d}.json') for r in rows}
    csv_data=pd.DataFrame.from_records(rows).to_csv(index=False,float_format='%.17g').encode('utf-8')
    fp=base/'predictions.csv'
    if not fp.is_file() or fp.read_bytes()!=csv_data:raise ValueError('FINAL_PREDICTIONS_CSV_DRIFT')
    path=base/'manifest.json'; m=json.loads(path.read_text('utf-8'))
    expected={'schema_version':'task011-final-forecast-manifest-v1',
       'status':'FINAL_FORECAST_CACHE_COMPLETE_NOT_SCIENTIFIC_CLEARANCE',
       'market':market,'model':model,'lock_sha256':lock_sha,'count':p['count'],
       'record_sha256':source_map,'predictions_sha256':sha256_file(fp),
       'checkpoint_id':identity[1],'checkpoint_commit':identity[2],
       'checkpoint_weight_sha256':identity[3],
       'context_length':256,'horizon':1,'samples':100,'quantiles':[.1,.5,.9],
       'seed_policy':'2027_plus_absolute_origin_index',**L.FLAGS}
    if m!=expected or path.read_bytes()!=L.canonical(m):
        raise ValueError('FINAL_MANIFEST_DRIFT')
    return rows,sha256_file(path)


def exclusive(lockfile:Path):
    """Minimal exclusive filesystem lock; no automatic stale-lock deletion."""
    from contextlib import contextmanager
    @contextmanager
    def _acquire():
        lockfile.parent.mkdir(parents=True,exist_ok=True)
        try:fd=os.open(lockfile,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:raise RuntimeError('FINAL_WORKER_LOCK_EXISTS_INSPECT_PID:'+str(lockfile)) from None
        try:
            os.write(fd,f'pid={os.getpid()}\n'.encode());os.close(fd)
            yield
        finally:lockfile.unlink(missing_ok=True)
    return _acquire()


def worker(root:Path,market:str,model:str,lock_sha:str):
    root=root.resolve()
    df,p=load_market_after_lock(root,market,lock_sha)
    identity=identity_from_lock(root,model,lock_sha)
    cls,model_id,rev,weight,files=identity
    from pilot_forecast_cache import verify_local_checkpoint
    if verify_local_checkpoint(model_id,rev,files)!=weight:raise ValueError('LOCAL_MODEL_WEIGHTS_DRIFT')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('NO_GPU_FOR_FROZEN_INFERENCE')
    base=market_dir(root,market,lock_sha,model)
    with exclusive(base/'.task011.lock'):
        prefix=read_records(root,market,model,lock_sha,df,p,identity)
        if (base/'manifest.json').exists():
            check_complete_manifest(root,market,model,lock_sha,df,p,identity)
            return {'status':'FINAL_CACHE_ALREADY_COMPLETE','market':market,'model':model,
                    'count':p['count'],'new_forecasts':0}
        adapter=None;made=0
        from cita2027.forecasting.adapters import seed_inference
        try:
            for j,o in enumerate(p['origins'][len(prefix):],start=len(prefix)):
                if adapter is None:
                    seed_inference(2027)
                    adapter=cls(revision=rev,device='cuda',samples=100)
                x=context_for_origin(df,o['origin_index'])
                seed_inference(o['seed'])
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                ts=time.perf_counter();q=adapter.predict(x)
                torch.cuda.synchronize()
                seconds=time.perf_counter()-ts
                check_quantiles(q)
                replay=j<2;diff=0.
                if replay:
                    seed_inference(o['seed']);q2=adapter.predict(x)
                    diff=float(np.max(np.abs(q-q2)))
                    if not np.allclose(q,q2,atol=1e-5,rtol=0):raise ValueError('FINAL_SEEDED_REPLAY_FAIL')
                # The realized test target is accessed ONLY after adapter.predict.
                observed=target_float(df,o['target_index'])
                r={'schema_version':'task011-final-forecast-v1','lock_sha256':lock_sha,
                    'market':market,'model':model,'checkpoint_id':model_id,
                    'checkpoint_commit':rev,'checkpoint_weight_sha256':weight,
                    'origin_index':o['origin_index'],'target_index':o['target_index'],
                    'origin_date':o['origin_date'],'target_date':o['target_date'],
                    'context_first_index':o['context_first_index'],
                    'context_last_index':o['context_last_index'],
                    'context_sha256':context_sha(x),'seed':o['seed'],
                    'context_length':256,'horizon':1,'samples':100,'quantile_levels':[.1,.5,.9],
                    'target_units':'daily_log_return_decimal',
                    'q10':float(q[0]),'q50':float(q[1]),'q90':float(q[2]),
                    'target_log_return':observed,'elapsed_seconds':float(seconds),
                    'peak_allocated_mib':float(torch.cuda.max_memory_allocated()/2**20),
                    'replay_checked':bool(replay),'replay_max_abs_diff':diff,**L.FLAGS}
                validate_record(r,o,df,lock_sha,market,model,identity)
                L.write_once(base/'records'/f'{o["origin_index"]:06d}.json',L.canonical(r))
                made+=1
                if made%50==0:print(json.dumps({'status':'FINAL_INFERENCE_PROGRESS',
                   'market':market,'model':model,'new_records':made,'total_complete':len(prefix)+made}),flush=True)
        finally:
            if adapter is not None:del adapter
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        rows=read_records(root,market,model,lock_sha,df,p,identity,complete=True)
        L.write_once(base/'predictions.csv',pd.DataFrame(rows).to_csv(index=False,float_format='%.17g').encode('utf-8'))
        manifest={'schema_version':'task011-final-forecast-manifest-v1',
         'status':'FINAL_FORECAST_CACHE_COMPLETE_NOT_SCIENTIFIC_CLEARANCE',
         'market':market,'model':model,'lock_sha256':lock_sha,'count':p['count'],
         'record_sha256':{f'{r["origin_index"]:06d}.json':sha256_file(base/'records'/f'{r["origin_index"]:06d}.json') for r in rows},
         'predictions_sha256':sha256_file(base/'predictions.csv'),
         'checkpoint_id':model_id,'checkpoint_commit':rev,'checkpoint_weight_sha256':weight,
         'context_length':256,'horizon':1,'samples':100,'quantiles':[.1,.5,.9],
         'seed_policy':'2027_plus_absolute_origin_index',**L.FLAGS}
        L.write_once(base/'manifest.json',L.canonical(manifest))
        check_complete_manifest(root,market,model,lock_sha,df,p,identity)
        return {'status':'FINAL_MODEL_FORECASTS_PASS','market':market,'model':model,
                'count':p['count'],'new_forecasts':made,
                'manifest_sha256':sha256_file(base/'manifest.json')}


def historical_records(root:Path,market:str,df:pd.DataFrame)->dict[str,list[dict]]:
    """Load already ACCEPTED CSV caches; do NOT run any old model or calibration."""
    import task007_freeze as spy_anchor
    import task009_freeze_synthesis as btc_anchor
    if market=='spy':
        blocks=[('full_development',L.SPY_DEV),('validation',L.SPY_VAL)]
        base=lambda kind,sha,model:root/'artifacts'/'forecast_cache'/kind/'runs'/sha/model
    else:
        blocks=[('development',L.BTC_DEV),('validation',L.BTC_VAL)]
        base=lambda kind,sha,model:root/'artifacts'/'forecast_cache'/'btcusdt'/kind/'runs'/sha/model
    result={}
    for model in L.MODELS:
        rows=[]
        for kind,sha in blocks:
            folder=base(kind,sha,model)
            expected=(spy_anchor.CACHE[(kind,model)][1] if market=='spy'
                      else btc_anchor.PINS[f'{"dev" if kind=="development" else "val"}_{model}_manifest'])
            if market=='spy':
                csv_hash=expected
                manifest_hash=spy_anchor.CACHE[(kind,model)][0]
            else:
                manifest_hash=expected
                manifest=json.loads((folder/'manifest.json').read_text('utf-8'))
                csv_hash=manifest['predictions_sha256']
            if sha256_file(folder/'manifest.json')!=manifest_hash or sha256_file(folder/'predictions.csv')!=csv_hash:
                raise ValueError('HISTORICAL_CACHE_HASH_DRIFT:'+market+':'+kind+':'+model)
            old=pd.read_csv(folder/'predictions.csv',float_precision='round_trip')
            expected_n=1258 if market=='spy' and kind=='full_development' else (
                502 if market=='spy' else (1461 if kind=='development' else 731))
            if len(old)!=expected_n:raise ValueError('ACCEPTED_CACHE_COUNT_DRIFT')
            rows.extend(old.to_dict('records'))
        indices=np.asarray([r['origin_index'] for r in rows],dtype=int)
        if not np.array_equal(indices,np.arange(indices[0],indices[0]+len(indices))):
            raise ValueError('ACCEPTED_HISTORICAL_ORIGIN_GAP')
        if (indices[0]!=(3272 if market=='spy' else 501)
            or indices[-1]!=(5031 if market=='spy' else 2692)):
            # SPY Validation ends at origin 5031; BTC Validation at 2692.
            raise ValueError('ACCEPTED_HISTORICAL_ORIGIN_BOUNDARY_DRIFT')
        if any(int(r['seed'])!=2027+int(r['origin_index']) for r in rows):
            raise ValueError('ACCEPTED_HISTORICAL_SEED_DRIFT')
        result[model]=rows
    return result


def compare_accepted_prefix(root:Path,market:str,model:str,cell:str,replayed:list[dict],n_dev:int):
    """State continuity: compare ALL accepted Development & Validation endpoints."""
    if market=='spy':
        pairs=[(f'artifacts/calibration/task005b/{L.SPY_DEV}/full_development_1006_per_cell.csv',
                '2f6f88345e5808404ddd609b535f64e3c6a64d0ab11c213305e0f9c73cf39e45',1006),
               (f'artifacts/calibration/validation/{L.SPY_VAL}/validation_502_per_cell.csv',
                None,502)]
        # SPY accepted Validation report records exact trace SHA.
        val_report=json.loads((root/f'artifacts/calibration/validation/{L.SPY_VAL}/validation_calibration_report.json').read_text('utf-8'))
        pairs[1]=(pairs[1][0],val_report['trace_sha256'],502)
    else:
        pairs=[(f'artifacts/calibration/btcusdt/development/{L.BTC_DEV}/development_1209_per_cell.csv',
                '19619db73f57d484ab902f551bef09741b506bca5a433b471b13ebe73a21e07d',1209),
               (f'artifacts/calibration/btcusdt/validation/{L.BTC_VAL}/validation_731_per_cell.csv',
                'a305b6024b434c57433a5e005c0dd3b232cfda9c86582a3fb9d50adf9a587201',731)]
    if len(replayed)!=sum(n for _,_,n in pairs):
        raise ValueError('HISTORICAL_CALIBRATION_OUTPUT_LENGTH_DRIFT')
    offset=0
    for rel,sha,expected in pairs:
        path=root/rel
        if sha256_file(path)!=sha:raise ValueError('ACCEPTED_CALIBRATION_TRACE_SHA_DRIFT')
        df=pd.read_csv(path,float_precision='round_trip')
        filtered=df[(df['model']==model)&(df['cell']==cell)].sort_values('origin_index')
        if len(filtered)!=expected:raise ValueError('HISTORICAL_CALIBRATION_TRACE_COUNT_DRIFT')
        for k in ('origin_index','target_index','lower','upper','scale','alpha_t','gamma_t',
                  'gamma_driver','return_volatility'):
            v=filtered[k].to_numpy()
            actual=np.asarray([r[k] for r in replayed[offset:offset+expected]])
            if k in ('origin_index','target_index'):
                if not np.array_equal(v,actual):raise ValueError('HISTORICAL_TRACE_ORIGIN_DRIFT:'+k)
            elif not np.allclose(v.astype(float),actual.astype(float),atol=1e-10,rtol=1e-10):
                raise ValueError('HISTORICAL_TRACE_CAUSAL_STATE_DRIFT:'+market+':'+model+':'+cell+':'+k)
        offset+=expected


def metric_check(m:dict,n:int):
    if m['n']!=n or abs(m['coverage']+m['lower_miss_rate']+m['upper_miss_rate']-1)>1e-10:
        raise ValueError('FINAL_METRIC_COUNT_OR_MISS_DECOMPOSITION_FAIL')
    if abs(m['winkler_mean']-(m['width_mean']+m['lower_penalty_mean']+m['upper_penalty_mean']))>1e-10:
        raise ValueError('FINAL_WINKLER_DECOMPOSITION_FAIL')
    for k in ('winkler_mean','coverage','width_mean','lower_miss_rate','upper_miss_rate'):
        if not np.isfinite(m[k]):raise ValueError('NONFINITE_FINAL_METRIC')


def centered_one_sided_p(d:np.ndarray,indices:np.ndarray)->float:
    """Pre-registered one-sided null H0 E[d]>=0; circular-block null recenter."""
    a=np.asarray(d,dtype=float)
    obs=float(a.mean())
    boot=a[indices].mean(axis=1)
    return (1+int(np.sum(boot-obs<=obs)))/(len(boot)+1)


def holm_adjust(pvalues:dict[str,float])->dict[str,float]:
    if not pvalues:raise ValueError('NO_PRIMARY_HYPOTHESES')
    if any(not 0<=p<=1 for p in pvalues.values()):raise ValueError('INVALID_PVALUES')
    ordered=sorted(pvalues,key=lambda k:(pvalues[k],k));m=len(ordered)
    result={};prev=0.
    for rank,key in enumerate(ordered):
        prev=max(prev,min(1.,(m-rank)*pvalues[key]))
        result[key]=prev
    return result


def final_regime_and_metric(rows:list[dict],mask:np.ndarray,draws:np.ndarray,params:Parameters):
    result={}
    vals=score_arrays(rows,params.alpha)
    for name,flag in [('high',mask),('other',~mask)]:
        subset=[r for r,f in zip(rows,flag) if f]
        if not subset:raise ValueError('EMPTY_FINAL_VOLATILITY_REGIME')
        m=interval_metrics(subset,params.alpha);metric_check(m,len(subset))
        result[name]={'n':len(subset),'metrics':m,
                      'block20_marginal_ci95':{key:masked_bootstrap_ci(v,flag,draws) for key,v in vals.items()}}
    return result


def score_from_trace(row):
    """Independent elementary recomputation used again by the file-level auditor."""
    lo=float(row['lower']);up=float(row['upper']);y=float(row['target_log_return'])
    if not np.isfinite([lo,up,y]).all() or lo>up:raise ValueError('INVALID_TRACE_ENDPOINT')
    return (up-lo)+10.0*max(lo-y,0.0)+10.0*max(y-up,0.0)


def evaluate(root:Path,lock_sha:str)->dict:
    root=root.resolve()
    lock,plans=L.load_lock(root,lock_sha)
    # First phase of evaluation: independent post-lock input/record integrity.
    frames={};past={};future={};mhash={}
    for market in ('spy','btcusdt'):
        frames[market],p=load_market_after_lock(root,market,lock_sha)
        past[market]=historical_records(root,market,frames[market])
        future[market]={}
        for model in L.MODELS:
            cls,mid,rev,weight,files=identity_from_lock(root,model,lock_sha)
            rr,sha=check_complete_manifest(root,market,model,lock_sha,frames[market],p,
                                           (cls,mid,rev,weight,files))
            future[market][model]=rr;mhash[f'{market}:{model}']=sha
        if [(r['origin_index'],r['target_date']) for r in future[market]['chronos']] != [
            (r['origin_index'],r['target_date']) for r in future[market]['moirai']]:
            raise ValueError('CROSS_MODEL_ORIGIN_ALIGNMENT_DRIFT')
        if any(float(a['target_log_return']).hex()!=float(b['target_log_return']).hex()
               for a,b in zip(future[market]['chronos'],future[market]['moirai'])):
            raise ValueError('CROSS_MODEL_REALIZED_TARGET_DRIFT')
        if past[market]['chronos'][-1]['target_index']!=future[market]['chronos'][0]['origin_index']:
            raise ValueError('VALIDATION_TO_FINAL_BOUNDARY_GAP')
    params=Parameters()
    if vars(params)!={'alpha':.2,'window':252,'decay':.94,'gamma':.01,'min_scale':1e-6,
                      'driver_min':.5,'driver_max':2.,'correction':'expansion_only'}:
        raise ValueError('FINAL_CALIBRATION_PARAMETERS_DRIFT')
    all_results={};primary={};trace=[];summary_rows=[];regime_rows=[]
    for market in ('spy','btcusdt'):
        n=plans[market]['count'];df=frames[market]
        first_i=3272 if market=='spy' else 501
        initial=context_for_origin(df,first_i)
        joint={model:past[market][model]+future[market][model] for model in L.MODELS}
        masks={}
        for model in L.MODELS:
            mask=causal_regime_indices(joint[model],initial)['high'][-n:]
            if len(mask)!=n or not mask.any() or mask.all():
                raise ValueError('FINAL_HIGH_OTHER_REGIME_EMPTY_OR_UNALIGNED')
            masks[model]=mask
        if not np.array_equal(masks['chronos'],masks['moirai']):
            raise ValueError('CROSS_MODEL_REGIME_MASK_DRIFT')
        mask=masks['chronos']
        idx={k:circular_block_indices(n,1000,k,2027) for k in (10,20,40)}
        regs={'high':int(mask.sum()),'other':int((~mask).sum())}
        all_results[market]={'n':n,'regime_counts':regs,'models':{}}
        ndev=1258 if market=='spy' else 1461
        for model in L.MODELS:
            cells={};series={}
            raw=[{'origin_index':r['origin_index'],'target_index':r['target_index'],
                  'origin_date':r['origin_date'],'target_date':r['target_date'],
                  'target_log_return':r['target_log_return'],'lower':r['q10'],'upper':r['q90']}
                  for r in future[market][model]]
            raw_s=score_arrays(raw,params.alpha)
            raw_m=interval_metrics(raw,params.alpha);metric_check(raw_m,n)
            all_results[market]['models'][model]={'raw_q10_q90':{'metrics':raw_m,
                               'regimes':final_regime_and_metric(raw,mask,idx[20],params)},'cells':{}}
            summary_rows.append({'market':market,'model':model,'cell':'raw_q10_q90',**raw_m})
            for regime in ('high','other'):
                mm=all_results[market]['models'][model]['raw_q10_q90']['regimes'][regime]['metrics']
                regime_rows.append({'market':market,'model':model,'cell':'raw_q10_q90',
                                    'regime':regime,**mm})
            for scale in SCALES:
                for ctrl in CONTROLLERS:
                    cell=f'{scale}__{ctrl}'
                    output=calibrate(joint[model],initial,params,scale,ctrl)
                    if len(output)!=len(joint[model])-252:
                        raise ValueError('CALIBRATOR_OUTPUT_LENGTH_DRIFT')
                    history=output[:len(past[market][model])-252]
                    compare_accepted_prefix(root,market,model,cell,history,ndev)
                    test=output[-n:]
                    expected_indices=[r['origin_index'] for r in future[market][model]]
                    if [r['origin_index'] for r in test]!=expected_indices:
                        raise ValueError('TEST_REPLAY_ORIGIN_DRIFT')
                    metrics=interval_metrics(test,params.alpha);metric_check(metrics,n)
                    cells[cell]=test;series[cell]=score_arrays(test,params.alpha)['winkler']
                    from cita2027.calibration.diagnostic_correction import gamma_diagnostics
                    diagnostics=gamma_diagnostics(pd.DataFrame([{'cell':cell,**r} for r in test]),params)
                    all_results[market]['models'][model]['cells'][cell]={
                        'metrics':metrics,'regimes':final_regime_and_metric(test,mask,idx[20],params),
                        'gamma_diagnostics':diagnostics}
                    summary_rows.append({'market':market,'model':model,'cell':cell,**metrics})
                    for regime,flag in (('high',mask),('other',~mask)):
                        mm=all_results[market]['models'][model]['cells'][cell]['regimes'][regime]['metrics']
                        regime_rows.append({'market':market,'model':model,'cell':cell,
                                            'regime':regime,**mm})
                    for row in test:
                        trace.append({'market':market,'model':model,'cell':cell,
                            'origin_index':row['origin_index'],'target_index':row['target_index'],
                            'origin_date':row['origin_date'],'target_date':row['target_date'],
                            'target_log_return':row['target_log_return'],'lower':row['lower'],
                            'upper':row['upper'],'winkler':score_from_trace(row),
                            'alpha_t':row['alpha_t'],'gamma_t':row['gamma_t'],
                            'gamma_driver':row['gamma_driver'],'scale':row['scale'],
                            'return_volatility':row['return_volatility'],
                            'residual_volatility':row['residual_volatility'],
                            'score_threshold':row['score_threshold']})
            for scale in SCALES:
                for ctrl in CONTROLLERS:
                    cell=f'{scale}__{ctrl}'
                    base=series['none__rolling_cqr']
                    fixed=series[f'{scale}__fixed_gamma_aci']
                    comparisons={'vs_unscaled_rolling':{},'vs_same_scale_fixed':{},'vs_raw_q10_q90':{}}
                    for name,comp in [('vs_unscaled_rolling',base),
                                      ('vs_same_scale_fixed',fixed),
                                      ('vs_raw_q10_q90',raw_s['winkler'])]:
                        d=series[cell]-comp
                        comparisons[name]={str(k):paired_ci(d,indices) for k,indices in idx.items()}
                    all_results[market]['models'][model]['cells'][cell]['paired']=comparisons
                if scale!='none':
                    cell=f'{scale}__rolling_cqr'
                    d=series[cell]-series['none__rolling_cqr']
                    key=f'{market}|{model}|{scale}'
                    primary[key]={'market':market,'model':model,'scale':scale,
                       'mean_difference':float(d.mean()),
                       'ci95':{str(k):paired_ci(d,indices)['ci_95_percentile'] for k,indices in idx.items()},
                       'p_uncorrected_one_sided_B20':centered_one_sided_p(d,idx[20]),
                       'n':n,'family':'eight_predeclared_normalization_contrasts'}
            # Keep signed-CQR explicitly diagnostic; never silently omit empty sets.
            history_scores=[signed_score(float(r['target_log_return']),float(r['q10']),float(r['q90'])) for r in joint[model]]
            good=[];bad=[]
            for j in range(len(past[market][model]),len(joint[model])):
                r=joint[model][j]
                q=conformal_order_statistic(history_scores[j-252:j],params.alpha)
                lo=float(r['q10'])-q;hi=float(r['q90'])+q
                if lo>hi or not np.isfinite([lo,hi]).all():bad.append(int(r['origin_index']))
                else:good.append({'lower':lo,'upper':hi,'target_log_return':r['target_log_return']})
            all_results[market]['models'][model]['signed_rolling_none']={
                'status':'EMPTY_SIGNED_SETS_WINKLER_UNDEFINED' if bad else 'FINITE_SIGNED_INTERVALS',
                'evaluated':n,'empty_count':len(bad),'empty_origins':bad,
                'coverage_with_empty_as_miss':(
                    sum(lo['lower']<=lo['target_log_return']<=lo['upper'] for lo in good)/n if bad else None),
                'metrics':interval_metrics(good,params.alpha) if not bad else None}
    if len(primary)!=8:raise ValueError('HOLM_PRIMARY_FAMILY_NOT_EIGHT')
    p_adj=holm_adjust({k:v['p_uncorrected_one_sided_B20'] for k,v in primary.items()})
    for key,item in primary.items():
        item['p_holm_adjusted_B20']=p_adj[key]
        item['holm_reject_alpha_0_05']=bool(p_adj[key]<=.05)
    target=out_dir(root,lock_sha)
    raw_trace=pd.DataFrame(trace).sort_values(['market','model','cell','origin_index'],kind='stable')
    trace_blob=raw_trace.to_csv(index=False,float_format='%.17g').encode('utf-8')
    trace_path=target/'final_per_origin_18cell.csv';L.write_once(trace_path,trace_blob)
    source_hashes={market:plans[market]['source'] for market in ('spy','btcusdt')}
    report={'schema_version':'task011-final-test-report-v1','task':'TASK-011',
       'status':'FINAL_TEST_CALIBRATION_COMPLETE_AUDIT_PENDING_NOT_SCIENTIFIC_CLEARANCE',
       'lock_sha256':lock_sha,'plan_sha256':{m:lock['plans'][m]['sha256'] for m in plans},
       'cached_forecast_manifest_sha256':mhash,'source_provenance':source_hashes,
       'parameters':vars(params),'calibration_state':'continuous_Development_Validation_Final_no_reset',
       'regime_rule':'return_EWMA_sigma_gt_prior_252_sigma_q75_causal',
       'bootstrap':{'blocks':[10,20,40],'reps':1000,'seed':2027,
                    'CI':'paired_circular_blocks_percentile_95_unadjusted',
                    'primary_test':'one_sided_centered_paired_circular_blocks_B20',
                    'holm_family':8,'holm_alpha':.05},
       'primary':primary,'markets':all_results,
       'per_origin_trace_sha256':sha256_file(trace_path),
       'native_and_nine_cells_per_model_market_verified':True,
       'tuning_after_validation':False,'test_method_selection_performed':False,
       'synthetic_bday_temporal_features_proven_equivalent':False,**L.FLAGS}
    report_path=target/'final_test_report.json';L.write_once(report_path,L.canonical(report))
    # Tables, descriptive only; no post-test picking of a single configuration.
    tab=target/'full_factorial_40_rows.csv'
    L.write_once(tab,pd.DataFrame(summary_rows).to_csv(index=False,float_format='%.17g').encode('utf-8'))
    reg=target/'high_other_regime_80_rows.csv'
    L.write_once(reg,pd.DataFrame(regime_rows).to_csv(index=False,float_format='%.17g').encode('utf-8'))
    primary_table=target/'primary_8_contrasts_holm.csv'
    L.write_once(primary_table,pd.DataFrame(primary.values()).to_csv(index=False).encode('utf-8'))
    return {'status':report['status'],'lock_sha256':lock_sha,
      'report_path':str(report_path),'report_sha256':sha256_file(report_path),
      'trace_sha256':sha256_file(trace_path),
      'primary_rejected_B20_holm':sum(x['holm_reject_alpha_0_05'] for x in primary.values()),
      'n_by_market':{m:plans[m]['count'] for m in ('spy','btcusdt')},
      'scientific_clearance':False}


def independent_audit_and_freeze(root:Path,lock_sha:str):
    """Read-only, non-inference post-execution evidence & basic independent score audit."""
    root=root.resolve();pre,plans=L.load_lock(root,lock_sha)
    out=out_dir(root,lock_sha)
    report_file=out/'final_test_report.json'
    report=json.loads(report_file.read_text('utf-8'))
    if report_file.read_bytes()!=L.canonical(report) or report['lock_sha256']!=lock_sha:
        raise ValueError('FINAL_REPORT_NOT_CANONICAL_OR_BAD_LOCK')
    rows=pd.read_csv(out/'final_per_origin_18cell.csv',float_precision='round_trip')
    if sha256_file(out/'final_per_origin_18cell.csv')!=report['per_origin_trace_sha256']:
        raise ValueError('FINAL_TRACE_HASH_DRIFT')
    expected_n=sum(plan['count'] for plan in plans.values())*2*9
    if len(rows)!=expected_n:raise ValueError('FINAL_TRACE_TOTAL_CARDINALITY_DRIFT')
    for market,p in plans.items():
        df,_=load_market_after_lock(root,market,lock_sha)
        for model in L.MODELS:
            cls,mid,rev,w,files=identity_from_lock(root,model,lock_sha)
            recs,manifest_hash=check_complete_manifest(root,market,model,lock_sha,df,p,
                                                      (cls,mid,rev,w,files))
            if manifest_hash!=report['cached_forecast_manifest_sha256'][f'{market}:{model}']:
                raise ValueError('FINAL_FORECAST_MANIFEST_SHA_DRIFT')
            expected_origin=[o['origin_index'] for o in p['origins']]
            for scale in SCALES:
                for ctrl in CONTROLLERS:
                    cell=f'{scale}__{ctrl}'
                    q=rows[(rows.market==market)&(rows.model==model)&(rows.cell==cell)]
                    q=q.sort_values('origin_index')
                    if len(q)!=p['count'] or list(q.origin_index)!=expected_origin:
                        raise ValueError('FINAL_TRACE_ORIGIN_DRIFT')
                    actual=np.array([score_from_trace(r) for r in q.to_dict('records')])
                    if not np.allclose(actual,q['winkler'].to_numpy(float),atol=1e-10,rtol=1e-10):
                        raise ValueError('TRACE_INDEPENDENT_WINKLER_RECALC_FAIL')
                    m=report['markets'][market]['models'][model]['cells'][cell]['metrics']
                    if abs(float(actual.mean())-m['winkler_mean'])>1e-10:
                        raise ValueError('TRACE_AGGREGATE_WINKLER_MISMATCH')
                    metric_check(m,p['count'])
                    for field in ('coverage','width_mean','lower_miss_rate','upper_miss_rate'):
                        arr=q[['lower','upper','target_log_return']].to_numpy(float)
                        lower,upper,y=arr[:,0],arr[:,1],arr[:,2]
                        check={
                          'coverage':np.mean((y>=lower)&(y<=upper)),
                          'width_mean':np.mean(upper-lower),
                          'lower_miss_rate':np.mean(y<lower),
                          'upper_miss_rate':np.mean(y>upper)}
                        if abs(float(check[field])-m[field])>1e-10:
                            raise ValueError('INDEPENDENT_TRACE_METRIC_MISMATCH:'+field)
    if set(report['primary'])!={f'{market}|{model}|{scale}' for market in ('spy','btcusdt') for model in L.MODELS for scale in ('residual_ewma','return_ewma')}:
        raise ValueError('HOLM_FAMILY_POSTHOC_CHANGE')
    if len(report['primary'])!=8 or report['bootstrap']['holm_family']!=8:
        raise ValueError('HOLM_NOT_PRE_REGISTERED')
    # Independent, read-only scientific audit of native metrics, regime
    # aggregation, every primary CI/p-value, and Holm familywise correction.
    # No model, calibrator or report is refitted/re-executed by this check.
    pvalues={}
    for market,plan in plans.items():
        draws={b:circular_block_indices(plan['count'],1000,b,2027) for b in (10,20,40)}
        for model in L.MODELS:
            r=report['markets'][market]['models'][model]
            df,_=load_market_after_lock(root,market,lock_sha)
            identity=identity_from_lock(root,model,lock_sha)
            recs,_=check_complete_manifest(root,market,model,lock_sha,df,plan,identity)
            native=[{'lower':v['q10'],'upper':v['q90'],
                     'target_log_return':v['target_log_return']} for v in recs]
            native_metrics=interval_metrics(native,.20)
            for field in ('winkler_mean','coverage','width_mean','lower_miss_rate','upper_miss_rate'):
                if abs(native_metrics[field]-r['raw_q10_q90']['metrics'][field])>1e-10:
                    raise ValueError('INDEPENDENT_NATIVE_METRIC_DRIFT:'+market+':'+model+':'+field)
            for cell,entry in [('raw_q10_q90',r['raw_q10_q90']),*r['cells'].items()]:
                all_m=entry['metrics']; high=entry['regimes']['high'];other=entry['regimes']['other']
                if high['n']+other['n']!=plan['count']:
                    raise ValueError('REGIME_COUNT_ARITHMETIC_DRIFT')
                for field in ('winkler_mean','coverage','width_mean','lower_miss_rate','upper_miss_rate'):
                    weighted=(high['n']*high['metrics'][field]+other['n']*other['metrics'][field])/plan['count']
                    if abs(weighted-all_m[field])>1e-10:
                        raise ValueError('REGIME_WEIGHTED_ARITHMETIC_DRIFT:'+market+':'+model+':'+cell)
            baseline=rows[(rows.market==market)&(rows.model==model)&
                          (rows.cell=='none__rolling_cqr')].sort_values('origin_index')
            if len(baseline)!=plan['count']:raise ValueError('PRIMARY_BASELINE_INCOMPLETE')
            for scale in ('residual_ewma','return_ewma'):
                test=rows[(rows.market==market)&(rows.model==model)&
                          (rows.cell==f'{scale}__rolling_cqr')].sort_values('origin_index')
                if list(test.origin_index)!=list(baseline.origin_index):
                    raise ValueError('PRIMARY_PAIRED_ORIGIN_MISALIGNMENT')
                delta=test.winkler.to_numpy(float)-baseline.winkler.to_numpy(float)
                key=f'{market}|{model}|{scale}'; entry=report['primary'][key]
                if abs(float(delta.mean())-entry['mean_difference'])>1e-10:
                    raise ValueError('PRIMARY_MEAN_NOT_REPRODUCIBLE:'+key)
                for b,draw in draws.items():
                    ci=paired_ci(delta,draw)['ci_95_percentile']
                    if not np.allclose(ci,entry['ci95'][str(b)],atol=1e-10,rtol=0):
                        raise ValueError('PRIMARY_CI_NOT_REPRODUCIBLE:'+key+':'+str(b))
                pv=centered_one_sided_p(delta,draws[20]);pvalues[key]=pv
                if abs(pv-entry['p_uncorrected_one_sided_B20'])>1e-10:
                    raise ValueError('PRIMARY_ONE_SIDED_P_NOT_REPRODUCIBLE:'+key)
    adjusted=holm_adjust(pvalues)
    for key,pv in adjusted.items():
        ent=report['primary'][key]
        if abs(pv-ent['p_holm_adjusted_B20'])>1e-10 or bool(pv<=.05)!=ent['holm_reject_alpha_0_05']:
            raise ValueError('HOLM_ADJUSTMENT_NOT_REPRODUCIBLE:'+key)
    inventory=[]
    for market in ('spy','btcusdt'):
        for model in L.MODELS:
            base=market_dir(root,market,lock_sha,model)
            manifest=json.loads((base/'manifest.json').read_text('utf-8'))
            files=[base/'manifest.json',base/'predictions.csv']+[
                base/'records'/name for name in sorted(manifest['record_sha256'])]
            for f in files:
                inventory.append({'path':f.relative_to(root).as_posix(),
                                  'sha256':sha256_file(f),'bytes':f.stat().st_size})
    for f in [out/'final_test_report.json',out/'final_per_origin_18cell.csv',
              out/'full_factorial_40_rows.csv',out/'high_other_regime_80_rows.csv',
              out/'primary_8_contrasts_holm.csv',out/'TASK-011_Results_Discussion_Addendum.md']:
        inventory.append({'path':f.relative_to(root).as_posix(),
                          'sha256':sha256_file(f),'bytes':f.stat().st_size})
    result={'schema_version':'task011-evidence-freeze-v1',
       'status':'FINAL_TEST_TECHNICAL_AUDIT_PASS_NOT_SCIENTIFIC_CLEARANCE',
       'lock_sha256':lock_sha,'final_report_sha256':sha256_file(report_file),
       'final_trace_sha256':report['per_origin_trace_sha256'],
       'final_records':sum(p['count'] for p in plans.values())*2,
       'reports_rechecked_against_traces':True,'accepted_pretest_anchor_sha256':lock_sha,
       'evidence':sorted(inventory,key=lambda e:e['path']),
       'contamination_free_verified':False,
       'point_in_time_adjusted_prices_verified':False,
       'btc_historical_point_in_time_verified':False,
       'scientific_clearance':False}
    blob=L.canonical(result);digest=L.sha_bytes(blob)
    path=out/f'evidence_{digest}.json';L.write_once(path,blob)
    return {'status':result['status'],'pretest_manifest_sha256':lock_sha,
      'final_evidence_sha256':digest,'final_evidence_path':str(path),
      'full_forecast_records_verified':result['final_records'],
      'indexed_files':len(inventory),'scientific_clearance':False}


def manuscript_update(root:Path,lock_sha:str):
    """Report *all* pre-registered contrasts; descriptive framing only."""
    out=out_dir(root,lock_sha)
    rep=json.loads((out/'final_test_report.json').read_text('utf-8'))
    table=['| Market | Model | Scale | Mean Δ Winkler | CI95 B20 | raw one-sided p | Holm p | Holm reject |',
      '|---|---|---|---:|---|---:|---:|---|']
    for k,ent in sorted(rep['primary'].items()):
        ci=ent['ci95']['20']
        table.append(f"| {ent['market']} | {ent['model']} | {ent['scale']} | {ent['mean_difference']:.7f} | [{ci[0]:.7f}, {ci[1]:.7f}] | {ent['p_uncorrected_one_sided_B20']:.5f} | {ent['p_holm_adjusted_B20']:.5f} | {ent['holm_reject_alpha_0_05']} |")
    notes='''# TASK-011 — Final Test Results & Discussion addendum (English)

The final-test protocol, target dates, model and calibration settings, comparison
family, Holm adjustment, and missing-data policy were locked before 2025–2026
realized targets were decoded by the execution pipeline. A SHA-256 pre-test
manifest was persisted before inference. Forecasts were generated using the
unchanged frozen Chronos-T5-small and Moirai-1.0-R-small adapters. Online
calibration states were carried forward by continuous chronological replay of
Development, Validation, and Final Test records, without reset or weight updates.

## Primary statistical comparisons

For each market and backbone, the two pre-registered comparisons measure the
paired mean Winkler difference between residual- or return-normalized Rolling
CQR and unscaled Rolling CQR. Negative differences favor normalization.
Unadjusted 95% paired circular moving-block bootstrap percentile intervals use
block lengths 10, 20, and 40. The primary confirmatory *decision procedure* is
a one-sided, centered bootstrap test at block length 20, with Holm adjustment
across all eight comparisons. Intervals are not familywise adjusted; they should
not be interpreted as such. Results are reported regardless of sign.

'''
    notes+='\n'.join(table)+'\n\n'
    notes+='''## Interpretation and limitations

Differences in mean interval score must be interpreted jointly with marginal
coverage, width, lower- and upper-tail misses, and volatility-regime diagnostics.
The fixed causal high-volatility regime indicator is based on pre-origin return
EWMA sigma compared with the preceding 252 origin sigmas; observed rolling
coverage is not a proof of conditional coverage. The eight pre-registered
comparisons control familywise error only under the bootstrap assumptions, not
against checkpoint training contamination or historical data revisions.
Chronos/Moirai test checkpoint training-data overlap remains unverified, Yahoo
adjusted close is retrospectively adjusted, Binance monthly archives may have
been corrected after their event times, and Moirai's synthetic business-day
calendar is not proven semantically equivalent to daily UTC BTC dynamics.
Native intervals and all nine factorial configurations remain disclosed.
Tests were not used to tune hyperparameters, select backbones, or revise the
pre-registered comparison family. The experiment is a controlled empirical
benchmark, not a newly proven conformal algorithm or a trading guarantee.

This addendum supplements TASK-010 and does not overwrite its accepted text.
Scientific clearance remains false.
'''
    path=out/'TASK-011_Results_Discussion_Addendum.md'
    L.write_once(path,notes.encode('utf-8'))
    return {'manuscript_addendum_path':str(path),'manuscript_sha256':sha256_file(path)}
