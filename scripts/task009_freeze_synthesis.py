"""TASK-009: read-only BTC evidence integrity audit and write-once science exports.

Uses the accepted BTC verification routines. Never invokes model inference/calibration,
network calls, or reads Final Test. Scientific flags are immutable false.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path

BTC_DEV = 'c8737fe917c830e6fb853343e684221d9f0c5543cd84807b92b79e7ee75aaf90'
BTC_VAL = '2a023e69a46aa952ca0015c2d8a87574616366a016c5cad333204418bda12e84'
DATA = '05088f11c7929995ef6dd28e5135669ebaf37465928c62c826a15f245520299a'
BOUNDARY = '48b9b79ff328441312330fb0b907582d3f2635d7d21e739390a3d41c55f10bb1'
SPY_DEV = '39c14b72d7bd82b815757147838e45fe3df56bddf4616367139cfd5661bd089f'
SPY_VAL = 'f574fd5be7e6537c263aa8c9a646dbb8676d87a25b38a70c83131cb0e894fe54'
# Accepted hashes from Windows execution and independent uploads (TASK-008F).
PINS = {
    'dev_report': '7782834dfa3aaa0e863a82a02c1e49604a493b2be2223467320fb782e0933210',
    'dev_comparison':'47f697cd1ea193d2fc0c78ba53e69158183cb15d612e44f5c89412bcc113060f',
    'dev_trace':'19619db73f57d484ab902f551bef09741b506bca5a433b471b13ebe73a21e07d',
    'dev_gate':'ec605263bd430a110e7bae6c5d11f9724c2ce4db75d7fa2daf79d2509c86c8ff',
    'dev_chronos_manifest':'d453a0233e841941277a5eb98ca78602202fedc6ed817a400bcf482454de7607',
    'dev_moirai_manifest':'26315b3cd3b9c589c308ab8ef78ccbdafe920c9b03f408a3e717482791b5121a',
    'val_report':'878cdd2880134a312561e036ced07528ff17a3a428ecda8e9a365e7de1457ee2',
    'val_comparison':'72d96bea3f7b797e7e115a10b70f07329c70c73d963182eb94c96ecb56f3f355',
    'val_trace':'a305b6024b434c57433a5e005c0dd3b232cfda9c86582a3fb9d50adf9a587201',
    'val_chronos_manifest':'df18911ce353480c31751b4abe1fd2557c23b8495bd66e933d751c6700a6e663',
    'val_moirai_manifest':'1307cbefacf130a8e53c7d2bbceb0f9b1c248b106c25e3f83a0f9e8b48392719',
    'spy_dev_report':'972a196f7beae4e31a343a1163beb678f76ee29e63362bb9b10171d3ea1ff149',
    'spy_val_report':'311429215bd96c0dbf52fcd0f758c28b6be4d5798941c3760210a158447828ed',
    'spy_comparison':'4deb6522ca69cbc6e4abc3f4e0fef9c1887d2d27b112712672be27081f868e31',
}
SOURCE_MANIFEST='76bbda3bef3be823324f237b5f95354ea55b0e75085a5d9d52b64351fe5f0f5f'
QUALITY='f5887a4d0554dd89a3338af12caa09e35f009a806f3ee097636e6caba22e790a'
MODELS=('chronos','moirai')
SCALES=('none','residual_ewma','return_ewma')
CONTROLLERS=('rolling_cqr','fixed_gamma_aci','variable_gamma_aci')
CELLS={f'{s}__{c}' for s in SCALES for c in CONTROLLERS}
FIELDS=('coverage','winkler_mean','width_mean','lower_miss_rate','upper_miss_rate')


def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def canonical(obj)->bytes:
    return (json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=True,allow_nan=False)+'\n').encode('utf-8')


def require(root:Path,rel:str,expected:str|None=None)->dict:
    p=root/rel
    if not p.is_file():raise ValueError('MISSING_ACCEPTED_ARTIFACT: '+rel)
    d=sha(p)
    if expected is not None and d!=expected:raise ValueError(f'ACCEPTED_FILE_SHA_DRIFT {rel} expected={expected} actual={d}')
    return {'path':rel,'sha256':d,'bytes':p.stat().st_size}


def load_pinned(root:Path,rel:str,expected:str)->dict:
    require(root,rel,expected)
    return json.loads((root/rel).read_text(encoding='utf-8'))


def metrics_good(metrics:dict,n:int):
    if metrics.get('n')!=n:raise ValueError('METRIC_COUNT_DRIFT')
    if abs(metrics['coverage']+metrics['lower_miss_rate']+metrics['upper_miss_rate']-1)>1e-10:
        raise ValueError('COVERAGE_PARTITION_DRIFT')
    if abs(metrics['winkler_mean'] - (metrics['width_mean']+metrics['lower_penalty_mean']+metrics['upper_penalty_mean']))>1e-10:
        raise ValueError('WINKLER_DECOMPOSITION_DRIFT')


def verify_report(report:dict,n:int,high:int,other:int):
    if set(report['models'])!=set(MODELS) or report['regime_counts']!={'high':high,'other':other}:
        raise ValueError('MODEL_REGIME_COUNT_DRIFT')
    for model in MODELS:
        rec=report['models'][model]
        if set(rec['cells'])!=CELLS:raise ValueError('MISSING_9_FACTORIAL_CELLS')
        metrics_good(rec['raw_q10_q90']['metrics'],n)
        for cell,entry in rec['cells'].items():
            m=entry['metrics'];metrics_good(m,n)
            for key,count in [('high',high),('other',other)]:
                r=entry['regimes'][key]
                if r['n']!=count:raise ValueError('REGIME_N_DRIFT')
                metrics_good(r['metrics'],count)
                for k in FIELDS:
                    combined=(high*entry['regimes']['high']['metrics'][k]+other*entry['regimes']['other']['metrics'][k])/n
                    if abs(combined-m[k])>1e-10:raise ValueError('REGIME_WEIGHTED_MEAN_DRIFT')
            for typ in ('vs_unscaled_rolling','vs_same_scale_fixed','vs_raw_q10_q90'):
                for block in ('10','20','40'):
                    c=entry['paired'][typ][block]
                    if c['n']!=n or not (c['ci_95_percentile'][0]<=c['ci_95_percentile'][1]):
                        raise ValueError('PAIRED_BOOTSTRAP_REPORT_DRIFT')
                    baseline=rec['raw_q10_q90']['metrics']['winkler_mean'] if typ=='vs_raw_q10_q90' else (
                        rec['cells']['none__rolling_cqr']['metrics']['winkler_mean'] if typ=='vs_unscaled_rolling' else
                        rec['cells'][f'{cell.split("__")[0]}__fixed_gamma_aci']['metrics']['winkler_mean'])
                    if abs(c['mean_difference']-(m['winkler_mean']-baseline))>1e-10:
                        raise ValueError('PAIRED_MEAN_MISMATCH')
    return True


def write_once(path:Path,data:bytes)->str:
    """Atomic publish via hard link; collision verified; never overwrite accepted bytes."""
    import tempfile
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('IMMUTABLE_OUTPUT_COLLISION: '+str(path))
        return sha(path)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.task009-',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        try:os.link(tmp,path)
        except FileExistsError:
            if path.read_bytes()!=data:raise ValueError('IMMUTABLE_OUTPUT_COLLISION: '+str(path))
        if sha(path)!=hashlib.sha256(data).hexdigest():raise ValueError('WRITEONCE_VERIFY_FAIL')
    finally:
        Path(tmp).unlink(missing_ok=True)
    return sha(path)


def csv_bytes(rows:list[dict])->bytes:
    buf=io.StringIO(newline='')
    writer=csv.DictWriter(buf,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    return buf.getvalue().encode('utf-8')


def load_sources(root:Path):
    dev_btc=load_pinned(root,f'artifacts/calibration/btcusdt/development/{BTC_DEV}/btc_development_calibration_report.json',PINS['dev_report'])
    val_btc=load_pinned(root,f'artifacts/calibration/btcusdt/validation/{BTC_VAL}/btc_validation_calibration_report.json',PINS['val_report'])
    dev_spy=load_pinned(root,f'artifacts/calibration/task005b/{SPY_DEV}/full_development_W252_expansion_only.json',PINS['spy_dev_report'])
    val_spy=load_pinned(root,f'artifacts/calibration/validation/{SPY_VAL}/validation_calibration_report.json',PINS['spy_val_report'])
    return {('BTC','Development'):dev_btc,('BTC','Validation'):val_btc,('SPY','Development'):dev_spy,('SPY','Validation'):val_spy}


def get_entry(source,dataset,period,model,cell):
    if dataset=='SPY':
        return source['models'][model]['cells'][cell] if period=='Development' else source['models'][model][cell]
    return source['models'][model]['cells'][cell]


def cell_metrics(source,dataset,period,model,cell):
    return get_entry(source,dataset,period,model,cell)['metrics']


def native(source,dataset,period,model):
    if dataset=='BTC':return source['models'][model]['raw_q10_q90']['metrics']
    return source['models'][model]['native_q10_q90']['metrics'] if period=='Development' else source['models'][model]['raw_q10_q90']['metrics']


def contrast(source,dataset,period,model,cell,typ,block):
    if dataset=='BTC':
        entry=get_entry(source,dataset,period,model,cell)['paired'][typ][str(block)]
        return entry['mean_difference'],entry['ci_95_percentile']
    if period=='Validation':
        d=get_entry(source,dataset,period,model,cell)['paired_contrasts'][{
            'vs_unscaled_rolling':'unscaled_rolling','vs_same_scale_fixed':'same_scale_fixed'}[typ]][str(block)]
        return d['mean'],d['ci95']
    # SPY Development paired normalization available from locked primary report; for
    # gamma vs fixed there is no paired bootstrap: denote CI unknown, not fabricated.
    if typ=='vs_unscaled_rolling' and cell.endswith('__rolling_cqr'):
        d=source['models'][model]['paired_vs_none_rolling'].get(cell)
        if d is not None:
            if block!=20:return d['mean_difference'],None
            return d['mean_difference'],d['ci_95_percentile']
    m=cell_metrics(source,dataset,period,model,cell)['winkler_mean']
    base=cell_metrics(source,dataset,period,model,f'{cell.split("__")[0]}__fixed_gamma_aci')['winkler_mean'] if typ=='vs_same_scale_fixed' else cell_metrics(source,dataset,period,model,'none__rolling_cqr')['winkler_mean']
    return m-base,None


def regime(source,dataset,period,model,cell,label):
    x=get_entry(source,dataset,period,model,cell)['regimes'][label]
    if dataset=='BTC':return x['metrics']
    if period=='Validation':
        return {'n':x['n'],'winkler_mean':x['winkler'],'coverage':x['covered'],
               'width_mean':x['width'],'lower_miss_rate':x['lower_miss'],
               'upper_miss_rate':x['upper_miss']}
    raise KeyError('SPY Development regime table not available in these inputs')


def create_tables(sources):
    main=[];pairs=[];reg=[]
    for (dataset,period),src in sources.items():
        for model in MODELS:
            keys=('raw_q10_q90',)+tuple(f'{s}__{c}' for s in SCALES for c in CONTROLLERS)
            for cell in keys:
                met=native(src,dataset,period,model) if cell=='raw_q10_q90' else cell_metrics(src,dataset,period,model,cell)
                main.append({'market':dataset,'period':period,'model':model,'configuration':cell,'n':met['n'],
                  'winkler':met['winkler_mean'],'coverage':met['coverage'],'width':met['width_mean'],
                  'lower_miss':met['lower_miss_rate'],'upper_miss':met['upper_miss_rate']})
            for scale in SCALES:
                for typ,controller in [('vs_unscaled_rolling','rolling_cqr'),('vs_same_scale_fixed','variable_gamma_aci')]:
                    if scale=='none' and typ=='vs_unscaled_rolling':continue
                    cell=f'{scale}__{controller}'
                    vals={b:contrast(src,dataset,period,model,cell,typ,b) for b in (10,20,40)}
                    means=[x[0] for x in vals.values()]
                    if max(means)-min(means)>1e-10:raise ValueError('CONTRAST_MEAN_DRIFT')
                    pairs.append({'market':dataset,'period':period,'model':model,'contrast':cell,'reference':typ,
                         'n':cell_metrics(src,dataset,period,model,cell)['n'],'delta_winkler':means[0],
                         **{f'ci{b}_lo':vals[b][1][0] if vals[b][1] else None for b in (10,20,40)},
                         **{f'ci{b}_hi':vals[b][1][1] if vals[b][1] else None for b in (10,20,40)},
                         'all_three_exclude_zero':None if any(vals[b][1] is None for b in (10,20,40)) else all(vals[b][1][1]<0 or vals[b][1][0]>0 for b in (10,20,40))})
            if period=='Validation':
                for cell in ('none__rolling_cqr','residual_ewma__rolling_cqr','return_ewma__rolling_cqr'):
                    for label in ('high','other'):
                        rr=regime(src,dataset,period,model,cell,label)
                        reg.append({'market':dataset,'period':period,'model':model,'configuration':cell,'regime':label,
                            'n':rr['n'],'winkler':rr['winkler_mean'],'coverage':rr['coverage'],
                            'width':rr['width_mean'],'lower_miss':rr['lower_miss_rate'],'upper_miss':rr['upper_miss_rate']})
    if len(main)!=80 or len(pairs)!=40 or len(reg)!=24:
        raise ValueError(f'INCOMPLETE_PUBLICATION_TABLES: {len(main)},{len(pairs)},{len(reg)}')
    return main,pairs,reg


def publish_synthesis(root:Path,sources,inventory_sha:str)->dict:
    main,pairs,reg=create_tables(sources)
    # Validate BTC and SPY report figures without executing any calibration.
    out=root/'artifacts'/'evidence'/'task009'/'tables'
    published={}
    for name,rows in [('table_full_factorial_80_rows.csv',main),('table_paired_contrasts_40_rows.csv',pairs),
                      ('table_validation_regimes_24_rows.csv',reg)]:
        data=csv_bytes(rows);published[name]=write_once(out/name,data)
    report={'task':'TASK-009','scope':'SPY_and_BTC_Development_Validation_aggregate_scientific_synthesis',
      'source_reports_sha256':{f'{market}_{period}':PINS['spy_dev_report' if market=='SPY' and period=='Development' else 'spy_val_report' if market=='SPY' else 'dev_report' if period=='Development' else 'val_report'] for market,period in sources},
      'btc_evidence_manifest_sha256':inventory_sha,'tables_sha256':published,
      'n_table_factorial':len(main),'n_table_contrasts':len(pairs),'n_table_validation_regimes':len(reg),
      'bootstrap_inference':'paired_within_each_market_period_only; CIs exploratory unadjusted for multiplicity',
      'spy_development_gamma_paired_ci':'NOT_AVAILABLE_IN_ACCEPTED_SOURCE_REPORT; leave blank',
      'no_cross_market_score_comparison':True,'no_cross_period_paired_test':True,
      'final_test_read':False,'contamination_free_verified':False,
      'btc_historical_point_in_time_verified':False,'point_in_time_adjusted_prices_verified':False,
      'scientific_clearance':False}
    content=canonical(report);digest=hashlib.sha256(content).hexdigest()
    write_once(root/'artifacts'/'evidence'/'task009'/'synthesis'/f'{digest}.json',content)
    return {'synthesis_sha256':digest,'table_hashes':published,'table_directory':str(out)}


def build_inventory(root:Path)->dict:
    # Existing accepted verifiers. NO GPU, no inference, no calibration, no SPY rerun.
    from cita2027.btc import development as dev, validation as val
    if (dev.DEV_SHA,dev.VAL_SHA,dev.DATA_SHA,val.BOUNDARY_SHA)!=(BTC_DEV,BTC_VAL,DATA,BOUNDARY):
        raise ValueError('FROZEN_SOURCE_CODE_PROTOCOL_DRIFT')
    df_dev,dev_plan,ident,_,_=dev.load_inputs(root)
    df_val,val_plan,ident_val,_,_,dev_rows,boundary=val.inputs(root)
    if len(dev_plan['origins'])!=1461 or len(val_plan['origins'])!=731 or boundary['development_forecasts_per_model']!=1461:
        raise ValueError('FROZEN_PLAN_RECORD_COUNT_DRIFT')
    inventory=[]
    sources={
      'artifacts/evidence/task007/23aac3bec7bafb6983f8f6678b8117f8175183d2f7596a67098e53bc070f6492.json':'23aac3bec7bafb6983f8f6678b8117f8175183d2f7596a67098e53bc070f6492',
      f'artifacts/btcusdt_data/snapshots/{DATA}.csv':DATA,
      f'artifacts/btcusdt_data/manifests/{SOURCE_MANIFEST}.json':SOURCE_MANIFEST,
      f'artifacts/btcusdt_data/quality/{QUALITY}.json':QUALITY,
      f'artifacts/btcusdt_data/plans/{BTC_DEV}.json':BTC_DEV,
      f'artifacts/btcusdt_data/plans/{BTC_VAL}.json':BTC_VAL,
      f'artifacts/btcusdt_boundary/task008d/{BOUNDARY}.json':BOUNDARY,
      f'artifacts/calibration/btcusdt/development/{BTC_DEV}/btc_development_calibration_report.json':PINS['dev_report'],
      f'artifacts/calibration/btcusdt/development/{BTC_DEV}/task008b_spy_btc_contrasts.json':PINS['dev_comparison'],
      f'artifacts/calibration/btcusdt/development/{BTC_DEV}/development_1209_per_cell.csv':PINS['dev_trace'],
      f'artifacts/calibration/btcusdt/validation/{BTC_VAL}/btc_validation_calibration_report.json':PINS['val_report'],
      f'artifacts/calibration/btcusdt/validation/{BTC_VAL}/task008e_dev_validation_comparison.json':PINS['val_comparison'],
      f'artifacts/calibration/btcusdt/validation/{BTC_VAL}/validation_731_per_cell.csv':PINS['val_trace'],
      f'artifacts/calibration/task005b/{SPY_DEV}/full_development_W252_expansion_only.json':PINS['spy_dev_report'],
      f'artifacts/calibration/validation/{SPY_VAL}/validation_calibration_report.json':PINS['spy_val_report'],
      f'artifacts/calibration/validation/{SPY_VAL}/task006b_scientific_comparison.json':PINS['spy_comparison'],
    }
    for rel,digest in sources.items():inventory.append(require(root,rel,digest))
    # TASK-008B gate file has content-hash as its name; path discovered in known BTC namespace only.
    inventory.append(require(root,f'artifacts/forecast_cache/btcusdt/development/gates/{BTC_DEV}.json',PINS['dev_gate']))
    for model in MODELS:
        if ident[model][1:4]!=ident_val[model][1:4]:raise ValueError('CHECKPOINT_IDENTITY_CHANGED_AT_VALIDATION')
        dev_manifest,dr=dev.verify_manifest(root,model,dev_plan,ident[model],df_dev)
        val_manifest,vr=val.check_manifest(root,model,val_plan,ident_val[model],df_val)
        if len(dr)!=1461 or len(vr)!=731 or dr[-1]['target_date']!='2022-12-31' or vr[0]['origin_date']!='2022-12-31' or vr[0]['target_date']!='2023-01-01':
            raise ValueError('BOUNDARY_OR_COUNT_DRIFT')
        for period,base,expected_manifest in [
            ('development',dev.cache_dir(root,model),PINS[f'dev_{model}_manifest']),
            ('validation',val.val_cache(root,model),PINS[f'val_{model}_manifest'])]:
            prefix=base.relative_to(root).as_posix()
            inventory.append(require(root,f'{prefix}/manifest.json',expected_manifest))
            manifest=dev_manifest if period=='development' else val_manifest
            inventory.append(require(root,f'{prefix}/predictions.csv',manifest['predictions_sha256']))
            names=manifest['record_sha256']
            if len(names)!=(1461 if period=='development' else 731):raise ValueError('RECORD_HASH_MAP_COUNT_DRIFT')
            actual={p.name for p in (base/'records').glob('*.json')}
            if actual!=set(names):raise ValueError('RECORD_SET_MISMATCH')
            for name,expected in sorted(names.items()):
                if Path(name).name!=name:raise ValueError('UNSAFE_RECORD_FILENAME')
                inventory.append(require(root,f'{prefix}/records/{name}',expected))
    dev.cross_model(dev_rows['chronos'],dev_rows['moirai'])
    val_rows={model:val.cache_records(root,model,val_plan,ident_val[model],df_val,complete=True) for model in MODELS}
    dev.cross_model(val_rows['chronos'],val_rows['moirai'])
    report_dev=load_pinned(root,f'artifacts/calibration/btcusdt/development/{BTC_DEV}/btc_development_calibration_report.json',PINS['dev_report'])
    report_val=load_pinned(root,f'artifacts/calibration/btcusdt/validation/{BTC_VAL}/btc_validation_calibration_report.json',PINS['val_report'])
    verify_report(report_dev,1209,284,925);verify_report(report_val,731,170,561)
    comparison=load_pinned(root,f'artifacts/calibration/btcusdt/validation/{BTC_VAL}/task008e_dev_validation_comparison.json',PINS['val_comparison'])
    if (comparison['validation_report_sha256']!=PINS['val_report'] or comparison['development_report_sha256']!=PINS['dev_report']
        or comparison['validation_plan_sha256']!=BTC_VAL or comparison['model_selection_performed'] is not False):
        raise ValueError('COMPARISON_REPORT_LINKAGE_DRIFT')
    for model in MODELS:
        for cell in CELLS:
            c=comparison['all_nine_factorial_cells'][model][cell]
            if (c['development_n'],c['validation_n'])!=(1209,731) or c['periods_not_paired'] is not True:
                raise ValueError('COMPARISON_PERIOD_PAIRING_DRIFT')
            if abs(c['validation_winkler']-report_val['models'][model]['cells'][cell]['metrics']['winkler_mean'])>1e-12:
                raise ValueError('COMPARISON_METRIC_DRIFT')
    for rel in ['src/cita2027/btc/development.py','src/cita2027/btc/validation.py',
                'src/cita2027/calibration/online.py','src/cita2027/forecasting/adapters.py',
                'scripts/task008b_btc_development.py','scripts/task008d_btc_boundary.py',
                'scripts/task008e_btc_validation.py']:
        inventory.append(require(root,rel))
    return {'schema_version':'task009-evidence-v1','task':'TASK-009',
      'status':'BTC_EVIDENCE_INDEX_VERIFIED_NOT_SCIENTIFIC_CLEARANCE',
      'source_dataset_sha256':DATA,'development_plan_sha256':BTC_DEV,'validation_plan_sha256':BTC_VAL,
      'boundary_state_report_sha256':BOUNDARY,'forecast_record_counts':{'development':2922,'validation':1462,'total':4384},
      'spy_evidence_index_sha256':'23aac3bec7bafb6983f8f6678b8117f8175183d2f7596a67098e53bc070f6492',
      'model_checkpoints':{m:{'checkpoint_id':ident[m][1],'revision':ident[m][2],'weight_sha256':ident[m][3]} for m in MODELS},
      'evidence':sorted(inventory,key=lambda item:item['path']),
      'record_hashing':'all_4384_json_records_checked_against_model_manifest_then_independently_sha256_indexed',
      'calibration_recomputed':False,'gpu_inference_performed':False,'final_test_read':False,
      'synthetic_bday_temporal_features_proven_equivalent':False,
      'contamination_free_verified':False,'btc_historical_point_in_time_verified':False,
      'point_in_time_adjusted_prices_verified':False,'scientific_clearance':False}


def run(root:Path):
    root=root.resolve()
    manifest=build_inventory(root)
    source=load_sources(root)
    # Source reports were pinned above; no SPY data re-audit.
    data=canonical(manifest)
    digest=hashlib.sha256(data).hexdigest()
    path=root/'artifacts'/'evidence'/'task009'/'btc'/f'{digest}.json'
    write_once(path,data)
    summary=publish_synthesis(root,source,digest)
    return {'status':manifest['status'],'btc_manifest_path':str(path),'btc_manifest_sha256':digest,
        'indexed_files':len(manifest['evidence']),'forecast_records_verified':4384,
        'development_forecasts':2922,'validation_forecasts':1462,**summary,
        'scientific_clearance':False,'contamination_free_verified':False,
        'btc_historical_point_in_time_verified':False,'point_in_time_adjusted_prices_verified':False,
        'final_test_read':False}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path('.'))
    a=ap.parse_args()
    print(json.dumps(run(a.root),sort_keys=True,indent=2))

if __name__=='__main__':main()
