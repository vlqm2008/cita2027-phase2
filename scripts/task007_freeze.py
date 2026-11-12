"""TASK-007: read-only verification of accepted SPY evidence, then write-once index.

No network, inference, calibration, model selection, or access to Final Test data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

DEV = '39c14b72d7bd82b815757147838e45fe3df56bddf4616367139cfd5661bd089f'
VAL = 'f574fd5be7e6537c263aa8c9a646dbb8676d87a25b38a70c83131cb0e894fe54'
EXPECTED = {
    f'artifacts/forecast_cache/full_development/plans/{DEV}.json': DEV,
    f'artifacts/forecast_cache/validation/plans/{VAL}.json': VAL,
    'artifacts/checkpoints/hf_audit.json': 'e426149cff7b3164793631276416ec10eb50b23b29048fb4b32e24ca35cbc974',
    'configs/smoke.yaml': 'a2047e99ee7447b8a7194064f7d4ba627288b23527255d5b770a9bb217aad0cf',
    f'artifacts/calibration/task005b/{DEV}/full_development_W252_expansion_only.json': '972a196f7beae4e31a343a1163beb678f76ee29e63362bb9b10171d3ea1ff149',
    f'artifacts/calibration/task005b/{DEV}/full_development_1006_per_cell.csv': '2f6f88345e5808404ddd609b535f64e3c6a64d0ab11c213305e0f9c73cf39e45',
    f'artifacts/calibration/task005b1/{DEV}/diagnostic_correction_v1.json': 'f461f1706a357726d1e3f5718697bca1e8770b4b868437604e12c20c330b1174',
    f'artifacts/calibration/validation/{VAL}/validation_calibration_report.json': '311429215bd96c0dbf52fcd0f758c28b6be4d5798941c3760210a158447828ed',
    f'artifacts/calibration/validation/{VAL}/task006b_scientific_comparison.json': '4deb6522ca69cbc6e4abc3f4e0fef9c1887d2d27b112712672be27081f868e31',
}
CACHE = {
 ('full_development','chronos'): ('c34ad91e66b4c9b0bc99500097baf6a61ea52d50bc12015ac6ca66e2239b07df', '671c1151895c1ce4718ab365a16b4abcc9b1c58b347ec05f6b607c9828a528f1',1258),
 ('full_development','moirai'): ('f6dcfea50973f6f3e568ecb8320e313f5232f83806c8671818da390b40f41a27', 'bdd0271729fd6fb47902193836d84d3745da3e128556c15a0aef4b476943f568',1258),
 ('validation','chronos'): ('ced314719eee5ce64185b8e16b05170cadad218ca1c8a84debdcced118ffdca3', 'f9ef1f691889b07d0bbf1955d9a815f81058bfc8ea3b24ba44e20cb673a0eac3',502),
 ('validation','moirai'): ('1fbad6312639c2b95141799bf29c4bb5a7f9c3ce2b080ff20a9c798c228b4e4b', '30df98645bf58714cc27c8bf560056927eb739d024f3403bab5396ddd66201b8',502),
}
CHECKPOINTS = {'chronos': {'id':'amazon/chronos-t5-small','revision':'a971ba21945c4f1796b17a91fe69214b5f4ad472','weight_sha256':'9c8b6fde5300f72b01c173153bf9288fa0a200614275bf0585071ad71a6a3d43'},
               'moirai': {'id':'Salesforce/moirai-1.0-R-small','revision':'f5bd5d01f0d67de856107d43abdd637380aae0a3','weight_sha256':'eaafeb166a3dfcf5728ab52a9ccc5392a23333b42a41a14e5528089ece3ef377'}}
FLAGS = ('contamination_free_verified', 'point_in_time_adjusted_prices_verified', 'scientific_clearance')

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def canonical(obj: dict) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)+'\n').encode()

def verify_path(root: Path, rel: str, expected: str) -> dict:
    path=root / rel
    if not path.is_file(): raise ValueError(f'MISSING_ACCEPTED_ARTIFACT: {rel}')
    actual=sha(path)
    if actual!=expected: raise ValueError(f'ARTIFACT_DRIFT: {rel}; expected {expected}, observed {actual}')
    return {'path':rel, 'sha256':actual,'size_bytes':path.stat().st_size}

def verify_cache(root: Path, kind: str, model: str, expected: tuple) -> list[dict]:
    manifest_sha, predictions_sha, count=expected
    plan = DEV if kind=='full_development' else VAL
    base=f'artifacts/forecast_cache/{kind}/runs/{plan}/{model}'
    items=[verify_path(root,f'{base}/manifest.json',manifest_sha),verify_path(root,f'{base}/predictions.csv',predictions_sha)]
    data=json.loads((root/base/'manifest.json').read_text('utf-8'))
    if data.get('count')!=count or data.get('model')!=model or data.get('plan_sha256')!=plan:
        raise ValueError(f'CACHE_META_DRIFT: {kind}/{model}')
    checked_flags = FLAGS if kind == 'validation' else FLAGS[:2]
    if any(data.get(flag) is not False for flag in checked_flags):
        raise ValueError('SAFETY_FLAG_DRIFT')
    expected_identity=CHECKPOINTS[model]
    if kind=='full_development':
        keys={'model_id':'id','model_revision':'revision','weight_sha256':'weight_sha256'}
    else:
        keys={'checkpoint_id':'id','checkpoint_commit':'revision','checkpoint_weight_sha256':'weight_sha256'}
    for key, dest in keys.items():
        if data.get(key)!=expected_identity[dest]:
            raise ValueError(f'CHECKPOINT_IDENTITY_DRIFT: {kind}/{model}/{key}')
    if data.get('predictions_sha256')!=predictions_sha:
        raise ValueError('CACHE_CSV_MANIFEST_DRIFT')
    record_map=data.get('record_sha256')
    if not isinstance(record_map,dict) or len(record_map)!=count:
        raise ValueError(f'MISSING_RECORD_HASH_MAP: {kind}/{model}')
    folder=root/base/'records'
    actual_names={p.name for p in folder.glob('*.json')}
    if actual_names!=set(record_map):
        raise ValueError(f'RECORD_SET_DRIFT: {kind}/{model}')
    # Deep verification of every accepted record: avoids trusting the manifest alone.
    for name, expected_digest in sorted(record_map.items()):
        if Path(name).name!=name or sha(folder/name)!=expected_digest:
            raise ValueError(f'RECORD_HASH_DRIFT: {kind}/{model}/{name}')
    items.append({'path':f'{base}/records/', 'count': count,
                  'integrity':'each_record_hash_matched_accepted_manifest',
                  'record_map_sha256':hashlib.sha256(canonical(record_map)).hexdigest()})
    return items

def build(root: Path) -> dict:
    items=[verify_path(root,rel,expected) for rel,expected in EXPECTED.items()]
    for (kind, model), values in CACHE.items(): items.extend(verify_cache(root,kind,model,values))
    # Verify accepted source dataset snapshot and its declared digest without reading test targets.
    dev=json.loads((root/f'artifacts/calibration/task005b/{DEV}/full_development_W252_expansion_only.json').read_text('utf-8'))
    if dev.get('processed_sha256')!='4375f3b90c5dbfe1cfd1a8191a36c15c2d878993ef04b130c8de8810927103ba':
        raise ValueError('PROCESSED_SNAPSHOT_IDENTITY_DRIFT')
    val=json.loads((root/f'artifacts/calibration/validation/{VAL}/validation_calibration_report.json').read_text('utf-8'))
    comp=json.loads((root/f'artifacts/calibration/validation/{VAL}/task006b_scientific_comparison.json').read_text('utf-8'))
    if any(obj.get(f) is not False for obj in (dev,val,comp) for f in FLAGS):
        raise ValueError('SCIENTIFIC_FLAG_DRIFT')
    if (dev.get('full_plan_sha256')!=DEV or val.get('validation_plan_sha256')!=VAL or comp.get('validation_plan_sha256')!=VAL
        or comp.get('validation_report_sha256')!=EXPECTED[f'artifacts/calibration/validation/{VAL}/validation_calibration_report.json']
        or val.get('records_per_model')!=502 or val.get('regime_counts')!={'high':71,'other':431}
        or comp.get('selection_performed') is not False or val.get('validation_hyperparameter_tuning') is not False
        or dev.get('final_test_used') is not False or val.get('final_test_used') is not False):
        raise ValueError('EVIDENCE_SEMANTIC_DRIFT')
    # Capture code/protocol versions as file checksums; do not assume files are git-committed.
    code=['scripts/task006a_validation.py','scripts/task006b_validation.py',
          'src/cita2027/calibration/online.py','src/cita2027/validation/boundary.py',
          'docs/task005a_math_protocol.md','docs/task005b1_diagnostic_protocol.md',
          'docs/task006a_validation_boundary_lock.md','docs/task006b_frozen_validation_execution.md',
          'configs/data.yaml']
    for rel in code:
        p=root/rel
        if not p.is_file(): raise ValueError(f'MISSING_PROTOCOL_SOURCE: {rel}')
        items.append({'path':rel,'sha256':sha(p),'size_bytes':p.stat().st_size})
    return {'schema_version':'task007-evidence-v1','task':'TASK-007','status':'EVIDENCE_INDEX_VERIFIED_NOT_SCIENTIFIC_CLEARANCE',
            'development_plan_sha256':DEV,'validation_plan_sha256':VAL,
            'scope':'SPY_warmup_development_validation_only',
            'model_checkpoints':CHECKPOINTS,
            'evidence':sorted(items,key=lambda e:e['path']),
            'verification':'deep_sha256_inventory_of_all_3520_forecast_records_and_reports',
            'final_test_read':False,'btc_downloaded':False,
            'contamination_free_verified':False,'point_in_time_adjusted_prices_verified':False,
            'scientific_clearance':False}

def publish(root:Path, report:dict) -> tuple[Path,str]:
    data=canonical(report); digest=hashlib.sha256(data).hexdigest()
    target=root/'artifacts'/'evidence'/'task007'/f'{digest}.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        with target.open('xb') as f: f.write(data);f.flush()
    except FileExistsError:
        if target.read_bytes()!=data: raise ValueError('EVIDENCE_FREEZE_COLLISION')
    if sha(target)!=digest: raise ValueError('EVIDENCE_FREEZE_WRITE_FAILED')
    return target,digest

def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('.'))
    args=ap.parse_args()
    root=args.root.resolve()
    report=build(root)
    path,digest=publish(root,report)
    print(json.dumps({'status':report['status'],'manifest_path':str(path),'manifest_sha256':digest,
      'indexed_items':len(report['evidence']),'records_hashed':3520,
      'scientific_clearance':False,'contamination_free_verified':False,
      'point_in_time_adjusted_prices_verified':False,'final_test_read':False},indent=2))

if __name__=='__main__':main()
