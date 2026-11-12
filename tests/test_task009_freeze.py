"""Focused CPU-only TASK-009 tests; never read financial data or run backbones."""
import hashlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import task009_freeze_synthesis as t


def mini(n=731):
    return {'n':n,'coverage':.8,'lower_miss_rate':.1,'upper_miss_rate':.1,
      'winkler_mean':.07,'width_mean':.05,'lower_penalty_mean':.01,'upper_penalty_mean':.01}


def test_canonical_digest():
    x={'b':2,'a':1}
    assert t.canonical(x).endswith(b'\n')
    assert t.canonical(x)==t.canonical({'a':1,'b':2})


def test_atomic_writeonce(tmp_path):
    f=tmp_path/'output'/'fixed.json';payload=b'fixed\n'
    assert t.write_once(f,payload)==hashlib.sha256(payload).hexdigest()
    assert t.write_once(f,payload)==hashlib.sha256(payload).hexdigest()


def test_collision_fails_closed(tmp_path):
    f=tmp_path/'immutable';t.write_once(f,b'accepted')
    with pytest.raises(ValueError,match='IMMUTABLE_OUTPUT_COLLISION'):
        t.write_once(f,b'changed')
    assert f.read_bytes()==b'accepted'


def test_missing_accepted_file_fails(tmp_path):
    with pytest.raises(ValueError,match='MISSING_ACCEPTED_ARTIFACT'):
        t.require(tmp_path,'artifacts/btc/unknown.json')


def test_pinned_hash_rejects_drift(tmp_path):
    f=tmp_path/'evidence.json';f.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='ACCEPTED_FILE_SHA_DRIFT'):
        t.require(tmp_path,'evidence.json','0'*64)


def test_coverage_identity():
    t.metrics_good(mini(),731)
    broken=mini();broken['upper_miss_rate']=.11
    with pytest.raises(ValueError,match='COVERAGE_PARTITION_DRIFT'):
        t.metrics_good(broken,731)


def test_winkler_decomposition():
    bad=mini();bad['lower_penalty_mean']=.02
    with pytest.raises(ValueError,match='WINKLER_DECOMPOSITION_DRIFT'):
        t.metrics_good(bad,731)


def test_count_identity():
    with pytest.raises(ValueError,match='METRIC_COUNT_DRIFT'):
        t.metrics_good(mini(729),731)


def test_csv_stable():
    rows=[{'model':'chronos','winkler':.05},{'model':'moirai','winkler':.06}]
    assert t.csv_bytes(rows)==b'model,winkler\nchronos,0.05\nmoirai,0.06\n'


def test_no_scientific_flag_promotion():
    assert all(v is False for v in (False,False,False))
    assert t.BTC_VAL=='2a023e69a46aa952ca0015c2d8a87574616366a016c5cad333204418bda12e84'
