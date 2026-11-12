"""Focused regression for TASK-011 identity schema. No GPU/network/Final Test."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from cita2027.finaltest import lock as L


def make_index():
    return {'model_checkpoints': {
        m:{'checkpoint_id':cfg['id'], 'revision':cfg['revision'],
           'weight_sha256':cfg['weight_sha256']}
        for m,cfg in L.CHECKPOINTS.items()}}


@pytest.mark.parametrize('model',list(L.MODELS))
def test_task009_schema_checked_without_weakening_identity(model):
    L.check_evidence_checkpoint_identity(make_index(),model)


@pytest.mark.parametrize('field', ['checkpoint_id','revision','weight_sha256'])
def test_each_identity_property_remains_strict(field):
    idx=make_index()
    idx['model_checkpoints']['chronos'][field]='MUTATED'
    with pytest.raises(ValueError,match='PINNED_MODEL_IDENTITY_DRIFT'):
        L.check_evidence_checkpoint_identity(idx,'chronos')


def test_missing_or_extra_keys_fail_closed():
    idx=make_index()
    del idx['model_checkpoints']['moirai']['revision']
    with pytest.raises(ValueError,match='SCHEMA_INVALID'):
        L.check_evidence_checkpoint_identity(idx,'moirai')
    idx=make_index()
    idx['model_checkpoints']['chronos']['unverified']='x'
    with pytest.raises(ValueError,match='SCHEMA_INVALID'):
        L.check_evidence_checkpoint_identity(idx,'chronos')


def test_actual_accepted_task009_manifest_format_if_present():
    # User's repository has this artifact; keep portability for clean offline
    # focused runs with no local evidence archive.
    root=Path(__file__).resolve().parents[1]
    p=root/'artifacts'/'evidence'/'task009'/'btc'/f'{L.BTC_INDEX}.json'
    if not p.is_file():
        pytest.skip('Accepted task009 manifest not mounted in offline fixture')
    idx=json.loads(p.read_text('utf-8'))
    for m in L.MODELS:
        L.check_evidence_checkpoint_identity(idx,m)
