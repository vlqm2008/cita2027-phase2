"""Metadata-only pre-test freeze. No realized 2025+ target is parsed here.

Bytes of existing snapshots/archives are hashed opaquely. Only timestamp/date
fields are decoded for coverage. Realized returns, prices, model quantiles and
metrics must not be inspected before the pretest manifest is durably published.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.error
import urllib.request
import zipfile

import pandas as pd

from cita2027.btc.development import DATA_SHA, DEV_SHA as BTC_DEV, VAL_SHA as BTC_VAL
from cita2027.validation.boundary import PROCESSED_SHA as SPY_DATA, DEV_PLAN_SHA as SPY_DEV
from cita2027.utils.provenance import sha256_file

SPY_VAL='f574fd5be7e6537c263aa8c9a646dbb8676d87a25b38a70c83131cb0e894fe54'
BTC_INDEX='718f0fcb8afeeff84ec00155cc621837d048dead55392264497f6c7f7ffefc96'
SPY_INDEX='23aac3bec7bafb6983f8f6678b8117f8175183d2f7596a67098e53bc070f6492'
SPY_VAL_REPORT='311429215bd96c0dbf52fcd0f758c28b6be4d5798941c3760210a158447828ed'
BTC_VAL_REPORT='878cdd2880134a312561e036ced07528ff17a3a428ecda8e9a365e7de1457ee2'
SPY_DEV_REPORT='972a196f7beae4e31a343a1163beb678f76ee29e63362bb9b10171d3ea1ff149'
BTC_DEV_REPORT='7782834dfa3aaa0e863a82a02c1e49604a493b2be2223467320fb782e0933210'
BOUNDARY='48b9b79ff328441312330fb0b907582d3f2635d7d21e739390a3d41c55f10bb1'
BASE='https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/'
START=date(2025,1,1)
END=date(2026,10,30)
MODELS=('chronos','moirai')
SCALES=('none','residual_ewma','return_ewma')
CONTROLLERS=('rolling_cqr','fixed_gamma_aci','variable_gamma_aci')
FLAGS={'contamination_free_verified':False,'point_in_time_adjusted_prices_verified':False,
       'btc_historical_point_in_time_verified':False,'scientific_clearance':False}
CHECKPOINTS={
 'chronos':{'id':'amazon/chronos-t5-small','revision':'a971ba21945c4f1796b17a91fe69214b5f4ad472',
            'weight_sha256':'9c8b6fde5300f72b01c173153bf9288fa0a200614275bf0585071ad71a6a3d43'},
 'moirai':{'id':'Salesforce/moirai-1.0-R-small','revision':'f5bd5d01f0d67de856107d43abdd637380aae0a3',
           'weight_sha256':'eaafeb166a3dfcf5728ab52a9ccc5392a23333b42a41a14e5528089ece3ef377'},
}


def canonical(obj)->bytes:
    return (json.dumps(obj,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')


def sha_bytes(blob:bytes)->str:return hashlib.sha256(blob).hexdigest()


def write_once(path:Path,blob:bytes):
    """Write-once publication via same-filesystem hard link; no overwrite on retry."""
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=blob:raise ValueError('IMMUTABLE_FILE_COLLISION:'+str(path))
        return sha_bytes(blob)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.partial-task011-',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(blob);f.flush();os.fsync(f.fileno())
        try:os.link(tmp,path)
        except FileExistsError:
            if path.read_bytes()!=blob:raise ValueError('RACE_IMMUTABLE_FILE_COLLISION:'+str(path))
    finally:Path(tmp).unlink(missing_ok=True)
    return sha_bytes(blob)


def pin(root:Path,rel:str,expected:str)->dict:
    p=root/rel
    if not p.is_file() or sha256_file(p)!=expected:
        raise ValueError('ACCEPTED_ANCHOR_MISSING_OR_SHA_DRIFT:'+rel)
    return {'path':rel,'sha256':expected,'bytes':p.stat().st_size}



def check_evidence_checkpoint_identity(index:dict, model:str)->None:
    """Compare locked BTC index schema to pinned local probe without weakening pins.

    TASK-009 index uses `checkpoint_id`, whereas TASK-011 local probe uses
    `id`. Only that property name is mapped; revision and SHA must be exact.
    """
    expected=CHECKPOINTS[model]
    recorded=index.get('model_checkpoints',{}).get(model)
    if not isinstance(recorded,dict) or set(recorded)!={'checkpoint_id','revision','weight_sha256'}:
        raise ValueError('PINNED_MODEL_IDENTITY_SCHEMA_INVALID:'+model)
    if (recorded['checkpoint_id']!=expected['id'] or
        recorded['revision']!=expected['revision'] or
        recorded['weight_sha256']!=expected['weight_sha256']):
        raise ValueError('PINNED_MODEL_IDENTITY_DRIFT:'+model)


def accepted_anchors(root:Path):
    paths={
      f'artifacts/evidence/task009/btc/{BTC_INDEX}.json':BTC_INDEX,
      f'artifacts/evidence/task007/{SPY_INDEX}.json':SPY_INDEX,
      'artifacts/checkpoints/hf_audit.json':'e426149cff7b3164793631276416ec10eb50b23b29048fb4b32e24ca35cbc974',
      'configs/smoke.yaml':'a2047e99ee7447b8a7194064f7d4ba627288b23527255d5b770a9bb217aad0cf',
      f'artifacts/btcusdt_data/plans/{BTC_DEV}.json':BTC_DEV,
      f'artifacts/btcusdt_data/plans/{BTC_VAL}.json':BTC_VAL,
      f'artifacts/btcusdt_data/snapshots/{DATA_SHA}.csv':DATA_SHA,
      f'artifacts/btcusdt_boundary/task008d/{BOUNDARY}.json':BOUNDARY,
      f'artifacts/forecast_cache/full_development/plans/{SPY_DEV}.json':SPY_DEV,
      f'artifacts/forecast_cache/validation/plans/{SPY_VAL}.json':SPY_VAL,
      f'artifacts/calibration/task005b/{SPY_DEV}/full_development_W252_expansion_only.json':SPY_DEV_REPORT,
      f'artifacts/calibration/validation/{SPY_VAL}/validation_calibration_report.json':SPY_VAL_REPORT,
      f'artifacts/calibration/btcusdt/development/{BTC_DEV}/btc_development_calibration_report.json':BTC_DEV_REPORT,
      f'artifacts/calibration/btcusdt/validation/{BTC_VAL}/btc_validation_calibration_report.json':BTC_VAL_REPORT,
    }
    listed=[pin(root,p,d) for p,d in sorted(paths.items())]
    idx=json.loads((root/f'artifacts/evidence/task009/btc/{BTC_INDEX}.json').read_text('utf-8'))
    if (idx.get('status')!='BTC_EVIDENCE_INDEX_VERIFIED_NOT_SCIENTIFIC_CLEARANCE' or
        idx.get('forecast_record_counts')!={'development':2922,'validation':1462,'total':4384} or
        idx.get('source_dataset_sha256')!=DATA_SHA or idx.get('validation_plan_sha256')!=BTC_VAL or
        idx.get('spy_evidence_index_sha256')!=SPY_INDEX or
        any(idx.get(k) is not False for k in FLAGS)):
        raise ValueError('ACCEPTED_EVIDENCE_INDEX_DRIFT')
    for model in MODELS:
        check_evidence_checkpoint_identity(idx, model)
    spy=json.loads((root/f'artifacts/evidence/task007/{SPY_INDEX}.json').read_text('utf-8'))
    if spy.get('status')!='EVIDENCE_INDEX_VERIFIED_NOT_SCIENTIFIC_CLEARANCE':
        raise ValueError('SPY_EVIDENCE_NOT_ACCEPTED')
    return listed


def spy_source_dates(root:Path):
    """Inspect ONLY source date/session labels; never parse adj_close/log_return."""
    from cita2027.validation.boundary import SOURCE_ID
    source=root/'artifacts'/'processed'/f'{SOURCE_ID}.csv'
    meta=root/'artifacts'/'manifests'/f'{SOURCE_ID}.json'
    if not meta.is_file() or not source.is_file():raise ValueError('SPY_PROCESSED_SOURCE_MISSING')
    metadata=json.loads(meta.read_text('utf-8'))
    if metadata.get('processed_sha256')!=SPY_DATA or sha256_file(source)!=SPY_DATA:
        raise ValueError('SPY_DATA_VINTAGE_DRIFT')
    # A CSV byte stream must be hashed; price/return fields are not decoded.
    dates=[]
    with source.open('r',encoding='utf-8',newline='') as f:
        head=next(csv.reader([f.readline()]))
        if 'date' not in head:raise ValueError('SPY_DATE_COLUMN_MISSING')
        date_pos=head.index('date')
        for line in f:
            parts=line.rstrip('\r\n').split(',')
            if len(parts)!=len(head):raise ValueError('SPY_CSV_ROW_SCHEMA_DRIFT')
            raw=parts[date_pos].strip('"')[:10]
            d=date.fromisoformat(raw)
            dates.append(d.isoformat())
    if not dates or dates!=sorted(set(dates)) or dates[5032]!='2024-12-31':
        raise ValueError('SPY_EXISTING_VALIDATION_BOUNDARY_DRIFT')
    final=[(i,d) for i,d in enumerate(dates) if START.isoformat()<=d<=END.isoformat()]
    if len(final)!=459 or final[0][0]!=5033 or final[-1][1]!=END.isoformat():
        raise ValueError('SPY_FINAL_ENDPOINT_OR_BOUNDARY_MISSING')
    if [i for i,_ in final]!=list(range(5033,5033+len(final))):
        raise ValueError('SPY_FINAL_INDEX_GAP')
    from exchange_calendars import get_calendar
    cal=get_calendar('XNYS')
    expected=[str(s.date()) for s in cal.sessions_in_range('2025-01-01',END.isoformat())]
    observed=[d for _,d in final]
    if observed!=expected:raise ValueError('SPY_XNYS_SESSION_GAP_OR_EXTRA')
    if dates[5032]!='2024-12-31' or observed[0]!='2025-01-02':
        raise ValueError('SPY_BRIDGE_DATE_MISMATCH')
    origins=[{'origin_index':i-1,'target_index':i,'origin_date':dates[i-1],
              'target_date':d,'context_first_index':i-256,'context_last_index':i-1,
              'seed':2027+i-1} for i,d in final]
    return {'source_path':source.relative_to(root).as_posix(), 'processed_sha256':SPY_DATA,
            'source_snapshot_id':SOURCE_ID,'metadata_path':meta.relative_to(root).as_posix(),
            'metadata_sha256':sha256_file(meta),'calendar':'XNYS','count':len(origins),
            'first_origin':'2024-12-31','first_target':observed[0], 'last_target':observed[-1]},origins


def archive_months():
    y,m=START.year,START.month
    while (y,m)<=(END.year,END.month):
        yield f'{y:04d}-{m:02d}'
        m+=1
        if m>12:y,m=y+1,1


def checksum_line(blob:bytes,name:str)->str:
    s=blob.decode('utf-8-sig').strip()
    match=re.fullmatch(r'([0-9a-fA-F]{64})\s+\*?([A-Za-z0-9_.-]+)',s)
    if not match or match.group(2)!=name:raise ValueError('BTC_OFFICIAL_CHECKSUM_FORMAT_DRIFT:'+name)
    return match.group(1).lower()


def fetch_bytes(url:str,max_bytes:int)->bytes:
    if not url.startswith(BASE) or not (url.endswith('.zip') or url.endswith('.zip.CHECKSUM')):
        raise ValueError('UNEXPECTED_SOURCE_URL')
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'CITA2027-task011-blind-freeze'}),timeout=45) as r:
            data=r.read(max_bytes+1)
    except (urllib.error.URLError,TimeoutError) as exc:
        raise RuntimeError('BLOCKER_BINANCE_ARCHIVE_UNAVAILABLE:'+url) from exc
    if len(data)>max_bytes:raise ValueError('EXCESSIVE_ARCHIVE_SIZE')
    return data


def date_labels_only(zip_bytes:bytes,month:str)->list[str]:
    """Ignore OHLC/returns; inspect UTC open/close time fields only."""
    name=f'BTCUSDT-1d-{month}.csv'
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        members=[x for x in z.infolist() if not x.is_dir()]
        if len(members)!=1 or members[0].filename!=name or members[0].file_size>64*1024*1024:
            raise ValueError('BTC_MONTH_ZIP_STRUCTURE_DRIFT:'+month)
        out=[]
        with z.open(name) as stream:
            for line in stream:
                # Only timestamps are decoded; all price/target fields remain opaque.
                p=line.strip().split(b',')
                if len(p)!=12:raise ValueError('BTC_MONTH_ROW_SCHEMA_DRIFT')
                if p[0].strip().lower() in (b'open_time',b'open time'):continue
                if not re.fullmatch(rb'\d{16}',p[0]) or not re.fullmatch(rb'\d{16}',p[6]):
                    raise ValueError('BTC_2025_MICROSECOND_TIMESTAMP_EXPECTED')
                op,cl=int(p[0]),int(p[6]);day_us=86_400_000_000
                if op%day_us!=0 or cl!=op+day_us-1:raise ValueError('BTC_UTC_CANDLE_TIMESTAMP_INVALID')
                d=datetime.fromtimestamp(op/1_000_000,timezone.utc).date().isoformat()
                if not d.startswith(month):raise ValueError('BTC_CANDLE_DATE_OUT_OF_MONTH')
                out.append(d)
    if out!=sorted(set(out)):raise ValueError('BTC_DUPLICATE_OR_UNORDERED_DATES')
    return out


def btc_source_dates(root:Path):
    cache=root/'artifacts'/'final_test'/'task011'/'blind_sources'/'binance_monthly'
    sources=[];whole=[]
    for month in archive_months():
        name=f'BTCUSDT-1d-{month}.zip'
        url=BASE+name
        z=cache/name; c=cache/(name+'.CHECKSUM')
        # Pretest raw bytes are immutable: never silently replace a cached archive.
        if c.exists() and z.exists():
            cs=c.read_bytes();blob=z.read_bytes()
        elif c.exists() or z.exists():
            raise ValueError('INCOMPLETE_PRETEST_ARCHIVE_PAIR:'+month)
        else:
            cs=fetch_bytes(url+'.CHECKSUM',4096)
            blob=fetch_bytes(url,64*1024*1024)
            if sha_bytes(blob)!=checksum_line(cs,name):raise ValueError('BTC_SOURCE_CHECKSUM_MISMATCH:'+month)
            write_once(c,cs);write_once(z,blob)
        target_hash=checksum_line(cs,name)
        if sha_bytes(blob)!=target_hash:raise ValueError('BTC_CACHED_ARCHIVE_DRIFT:'+month)
        labels=date_labels_only(blob,month)
        if not labels:raise ValueError('EMPTY_BINANCE_MONTH:'+month)
        whole.extend(labels)
        sources.append({'month':month,'path':z.relative_to(root).as_posix(),
                        'sha256':target_hash,'checksum_path':c.relative_to(root).as_posix(),
                        'checksum_sha256':sha_bytes(cs),'days':len(labels),
                        'timestamp_unit':'microseconds','date_first':labels[0],'date_last':labels[-1]})
    expected=[(START+timedelta(days=j)).isoformat() for j in range((END-START).days+1)]
    # Oct 2026 monthly archive has 31st; lock only targets through Oct 30.
    expected_extra=expected+[date(2026,10,31).isoformat()]
    if whole!=expected_extra:raise ValueError('BTC_UTC_DAY_GAP_OR_EXTRA')
    first_i=2694
    origins=[];before=date(2024,12,31).isoformat()
    for j,d in enumerate(expected):
        i=first_i+j
        origins.append({'origin_index':i-1,'target_index':i,'origin_date':before,
                        'target_date':d,'context_first_index':i-256,'context_last_index':i-1,
                        'seed':2027+i-1})
        before=d
    if len(origins)!=668 or origins[0]['origin_date']!='2024-12-31' or origins[0]['target_date']!='2025-01-01':
        raise ValueError('BTC_2025_BOUNDARY_DRIFT')
    return {'source':'Binance public BTCUSDT spot 1d monthly archive',
            'archives':sources,'calendar':'UTC_7day_including_weekends',
            'accepted_prefix_sha256':DATA_SHA,'count':len(origins),'first_origin':'2024-12-31',
            'first_target':START.isoformat(),'last_target':END.isoformat()},origins


def pinned_code(root:Path):
    # The accepted TASK-009 manifest already inventories these old source hashes.
    idx=json.loads((root/f'artifacts/evidence/task009/btc/{BTC_INDEX}.json').read_text('utf-8'))
    source_map={e['path']:e['sha256'] for e in idx['evidence']}
    old=['src/cita2027/forecasting/adapters.py','src/cita2027/calibration/online.py',
         'src/cita2027/btc/development.py','src/cita2027/btc/validation.py',
         'scripts/task008d_btc_boundary.py','scripts/task008e_btc_validation.py']
    # Also freeze imported scoring, regime and accepted data/plan reader implementations.
    # These bytes are from the previously accepted local-source packages, not
    # inferred from the new Final Test outcomes. Any drift is a fail-closed gate.
    mandatory={
      'src/cita2027/calibration/development_analysis.py':'4e847711132e8640cf0fc0997b78837c918ee68770db54343d582f0094780620',
      'src/cita2027/calibration/diagnostic_correction.py':'068c7a96e367968d538983c2c91f92f3658077bbfd17198c696d2098a730ad9d',
      'src/cita2027/forecasting/protocol.py':'d05ce37c225ac300466d06de73349d58e3978085fc7fed98e050e44e5788bbeb',
      'src/cita2027/btc/ingest.py':'3e5e854848c0e39226e808d96cdc47705f96298e88163b83c053f11783de0a3a',
      'src/cita2027/validation/boundary.py':'b0c9d194f1c6cd823834e8f724e0a20de02427c2e011ff1f4f89c97dfe2ed616',
      'src/cita2027/forecasting/cache.py':'508911e9100f1de1e3c808063064d397267e4c470e949f9a635fa9ae6edcd468',
    }
    return [pin(root,p,source_map[p]) for p in old]+[pin(root,p,h) for p,h in mandatory.items()]


def locked_rules():
    return {
      'target_start':'2025-01-01','target_end_inclusive':'2026-10-30',
      'markets':['spy','btcusdt'],'backbones':list(MODELS),
      'context':256,'horizon':1,'samples':100,'quantiles':[.1,.5,.9],
      'seed':'2027_plus_absolute_origin_index',
      'normalization':list(SCALES),'controllers':list(CONTROLLERS),
      'window':252,'nominal_alpha':.20,'gamma0':.01,'ewma_decay':.94,
      'min_scale':1e-6,'gamma_driver':'return_ewma','gamma_driver_bounds':[.5,2.],
      'correction':'expansion_only','ACI_alpha_clip':'[1/(W+1),1-1/(W+1)]',
      'missing_data':'FAIL_CLOSED_no_drop_no_impute_no_substitution',
      'forecast_failure':'HALT_resume_same_origin_no_skip',
      'incomplete_report':'NO_METRICS_UNTIL_ALL_FORECASTS_COMPLETE',
      'primary_hypotheses':[{'market':market,'model':model,'scale':scale,
         'A':f'{scale}__rolling_cqr','B':'none__rolling_cqr',
         'H1':'mean_Winkler(A)-mean_Winkler(B)<0'}
         for market in ('spy','btcusdt') for model in MODELS
         for scale in ('residual_ewma','return_ewma')],
      'primary_metric':'paired_mean_Winkler_difference_decimal_log_return',
      'primary_test':'one_sided_centered_paired_circular_moving_block_bootstrap',
      'primary_block_length':20,'bootstrap_sensitivity':[10,40],
      'bootstrap_replicates':1000,'bootstrap_seed':2027,
      'primary_family':'8_comparisons_Holm_Bonferroni_alpha_0.05_for_B20',
      'ci':'unadjusted_two_sided_percentile_95_B10_B20_B40',
      'secondary':'variable_gamma_vs_fixed_gamma_and_regime_effects_exploratory_unadjusted',
      'reported':'all_9_cells_per_model_market_plus_native_q10_q90_and_signed_CQR_diagnostic',
      'regimes':'causal_return_EWMA_sigma_gt_prior_252_sigma_q75',
      'rolling_coverage_window':50,'comparison_policy':'no_validation_or_test_based_model_selection',
      'statistical_scope':'within_market_model_origin_paired_no_cross_market_paired_test',
      'pretrained_model_overlap_proven_absent':False,
      'SPY_adj_close_point_in_time_proven':False,
      'BTC_archives_contemporaneously_available_proven':False,
      'Moirai_synthetic_BDay_7day_equivalent_proven':False,
      **FLAGS}


def task011_code_hashes(root:Path)->list[dict]:
    files=['src/cita2027/finaltest/lock.py','src/cita2027/finaltest/execute.py',
           'src/cita2027/finaltest/__init__.py','scripts/task011_final_test.py']
    return [{'path':p,'sha256':sha256_file(root/p)} for p in files]


def preflight_and_lock(root:Path,checkpoint_probes:dict):
    root=root.resolve()
    accepted=accepted_anchors(root)
    accepted+=pinned_code(root)
    if set(checkpoint_probes)!=set(MODELS) or any(checkpoint_probes[m]!=CHECKPOINTS[m] for m in MODELS):
        raise ValueError('PHASE_A_LOCAL_CHECKPOINT_SHA_PROBE_NOT_VERIFIED')
    spy,spy_origins=spy_source_dates(root)
    btc,btc_origins=btc_source_dates(root)
    # Include unambiguous source identities and exact date-only origin lists in lock.
    plans={}
    for market,source,origins in (('spy',spy,spy_origins),('btcusdt',btc,btc_origins)):
        item={'schema':'task011-final-origin-plan-v1','market':market,
              'origin_list_blind_date_metadata_only':True,
              'source':source,'count':len(origins),'origins':origins,
              'development_plan_sha256':SPY_DEV if market=='spy' else BTC_DEV,
              'validation_plan_sha256':SPY_VAL if market=='spy' else BTC_VAL,
              'context_hashes_deferred_until_after_freeze':True,
              **FLAGS}
        blob=canonical(item);digest=sha_bytes(blob)
        p=root/'artifacts'/'final_test'/'task011'/'locked'/'plans'/f'{market}_{digest}.json'
        write_once(p,blob)
        plans[market]={'path':p.relative_to(root).as_posix(),'sha256':digest,'count':len(origins)}
    # Tool cannot reveal adjusted return values or any test scoring before this barrier.
    spec={'schema':'task011-pretest-manifest-v1',
          'status':'PRETEST_PROTOCOL_FROZEN_NO_TEST_TARGET_PARSED',
          'task':'TASK-011','plans':plans,'accepted_anchors':accepted,
          'checkpoints':CHECKPOINTS,'task011_code_sha256':task011_code_hashes(root),
          'checkpoint_local_weight_verified_in_separate_envs':checkpoint_probes,
          'reporting_rules':locked_rules(),
          'source_coverage':'SPY_XNYS_and_BTC_UTC_exact_date_lists_verified',
          'source_prelock_access':'opaque_SHA256_and_date_timestamp_fields_only_no_prices_no_returns',
          'model_or_calibration_run_before_lock':False,'final_test_target_values_decoded_before_lock':False,
          'unresolved_provenance':list(FLAGS),**FLAGS}
    blob=canonical(spec);digest=sha_bytes(blob)
    path=root/'artifacts'/'final_test'/'task011'/'locked'/f'{digest}.json'
    write_once(path,blob)
    # Verify durable content under its hash BEFORE any execution code may be called.
    if sha_bytes(path.read_bytes())!=digest:raise ValueError('PRETEST_MANIFEST_PUBLICATION_FAILED')
    return {'status':'PHASE_A_PRETEST_LOCK_PASS_NOT_SCIENTIFIC_CLEARANCE',
            'pretest_manifest_sha256':digest,'pretest_manifest_path':str(path),
            'plans':plans,'test_targets_decoded_before_freeze':False,'scientific_clearance':False}


def load_lock(root:Path,digest:str):
    p=root/'artifacts'/'final_test'/'task011'/'locked'/f'{digest}.json'
    if not re.fullmatch('[0-9a-f]{64}',digest) or not p.is_file() or sha256_file(p)!=digest:
        raise ValueError('PRETEST_LOCK_NOT_DURABLE_OR_SHA_MISMATCH')
    lock=json.loads(p.read_text('utf-8'))
    if lock.get('status')!='PRETEST_PROTOCOL_FROZEN_NO_TEST_TARGET_PARSED' or lock['reporting_rules']!=locked_rules():
        raise ValueError('PRETEST_PROTOCOL_CHANGED_AFTER_LOCK')
    for a in lock['accepted_anchors']:pin(root,a['path'],a['sha256'])
    for a in lock['task011_code_sha256']:pin(root,a['path'],a['sha256'])
    plans={}
    for market in ('spy','btcusdt'):
        item=lock['plans'][market]
        rel=item['path'];pin(root,rel,item['sha256'])
        obj=json.loads((root/rel).read_text('utf-8'))
        if obj['market']!=market or obj['count']!=len(obj['origins']) or obj['count']!=item['count']:
            raise ValueError('PRETEST_PLAN_SCOPE_DRIFT')
        plans[market]=obj
    return lock,plans
