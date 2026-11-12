"""TASK-007: focused read-only evidence freeze safety tests."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'scripts'))
from task007_freeze import canonical, publish, verify_path, verify_cache, CHECKPOINTS

def test_canonical_stable():
    assert canonical({'b':2,'a':1})==canonical({'a':1,'b':2})

def test_write_once_idempotence(tmp_path):
    p,h=publish(tmp_path,{'status':'ok'});p2,h2=publish(tmp_path,{'status':'ok'})
    assert p==p2 and h==h2
    with pytest.raises(FileExistsError): p.open('xb')

def test_hash_mismatch_fails(tmp_path):
    p=tmp_path/'accepted';p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='ARTIFACT_DRIFT'):
        verify_path(tmp_path,'accepted','0'*64)

def test_missing_fails(tmp_path):
    with pytest.raises(ValueError, match='MISSING_ACCEPTED'):
        verify_path(tmp_path,'accepted','0'*64)

def test_tamper_fails_validation_record(tmp_path):
    # Construct a tiny standalone analogue; verify uses exact count and record map.
    from task007_freeze import canonical,sha,DEV,VAL
    base=tmp_path/f'artifacts/forecast_cache/validation/runs/{VAL}/chronos'
    (base/'records').mkdir(parents=True)
    (base/'records'/'000001.json').write_bytes(b'original')
    (base/'predictions.csv').write_bytes(b'csv')
    manifest={'count':1,'model':'chronos','plan_sha256':VAL,
      'checkpoint_id':CHECKPOINTS['chronos']['id'],
      'checkpoint_commit':CHECKPOINTS['chronos']['revision'],
      'checkpoint_weight_sha256':CHECKPOINTS['chronos']['weight_sha256'],
      'predictions_sha256':sha(base/'predictions.csv'),
      'record_sha256':{'000001.json':sha(base/'records'/'000001.json')},
      'contamination_free_verified':False,'point_in_time_adjusted_prices_verified':False,
      'scientific_clearance':False}
    (base/'manifest.json').write_bytes(canonical(manifest))
    pair=(sha(base/'manifest.json'),sha(base/'predictions.csv'),1)
    assert len(verify_cache(tmp_path,'validation','chronos',pair))==3
    (base/'records'/'000001.json').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='RECORD_HASH_DRIFT'):
        verify_cache(tmp_path,'validation','chronos',pair)
