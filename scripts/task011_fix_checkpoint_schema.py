"""Fail-closed TASK-011 Phase-A model identity schema hotfix.

This tool changes ONLY the new, not-yet-accepted TASK-011 lock.py source. It
never opens Final Test targets, revises a checkpoint, or changes old artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

ORIGINAL_SHA = '94dbac2dc60656c055150cc7ca78e5a106465f4d31eb17ab9554cb7b00fba91f'
FIXED_SHA = 'b3271354fa853e372b04c205ffa097bbc54e284997147c48cd5bb8f6b37ce00b'
OLD_LOOP = b"""    for model in MODELS:\n        if idx['model_checkpoints'][model]!=CHECKPOINTS[model]:\n            raise ValueError('PINNED_MODEL_IDENTITY_DRIFT:'+model)\n"""
NEW_LOOP = b"""    for model in MODELS:\n        check_evidence_checkpoint_identity(idx, model)\n"""
HEADER = b"""

def check_evidence_checkpoint_identity(index:dict, model:str)->None:
    \"\"\"Compare locked BTC index schema to pinned local probe without weakening pins.

    TASK-009 index uses `checkpoint_id`, whereas TASK-011 local probe uses
    `id`. Only that property name is mapped; revision and SHA must be exact.
    \"\"\"
    expected=CHECKPOINTS[model]
    recorded=index.get('model_checkpoints',{}).get(model)
    if not isinstance(recorded,dict) or set(recorded)!={'checkpoint_id','revision','weight_sha256'}:
        raise ValueError('PINNED_MODEL_IDENTITY_SCHEMA_INVALID:'+model)
    if (recorded['checkpoint_id']!=expected['id'] or
        recorded['revision']!=expected['revision'] or
        recorded['weight_sha256']!=expected['weight_sha256']):
        raise ValueError('PINNED_MODEL_IDENTITY_DRIFT:'+model)

"""


def sha(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def fix(root:Path)->dict:
    root=root.resolve()
    p=root/'src'/'cita2027'/'finaltest'/'lock.py'
    if not p.is_file():
        raise ValueError('TASK011_LOCK_SOURCE_MISSING:'+str(p))
    raw=p.read_bytes()
    digest=sha(raw)
    if digest==FIXED_SHA:
        return {'status':'TASK011_IDENTITY_SCHEMA_HOTFIX_ALREADY_APPLIED',
                'sha256':digest,'final_test_read':False,'checkpoints_modified':False}
    if digest!=ORIGINAL_SHA:
        raise ValueError('TASK011_LOCK_SOURCE_SHA_UNRECOGNIZED_STOP_NO_WRITE:'+digest)
    # A published pre-test lock includes source-code hashes; do not alter it.
    locked_dir=root/'artifacts'/'final_test'/'task011'/'locked'
    if locked_dir.is_dir() and any(re.fullmatch(r'[0-9a-f]{64}\.json',x.name) for x in locked_dir.iterdir() if x.is_file()):
        raise ValueError('EXISTING_PRETEST_MANIFEST_STOP_NO_CODE_CHANGE')
    marker=b'\ndef accepted_anchors(root:Path):'
    if raw.count(OLD_LOOP)!=1 or raw.count(marker)!=1:
        raise ValueError('TASK011_LOCK_SCHEMA_SOURCE_ANCHOR_NOT_UNIQUE')
    corrected=raw.replace(marker,HEADER+marker).replace(OLD_LOOP,NEW_LOOP)
    if sha(corrected)!=FIXED_SHA:
        raise ValueError('HOTFIX_OUTPUT_SHA_UNEXPECTED_STOP_NO_WRITE')
    fd,tmp=tempfile.mkstemp(dir=p.parent,prefix='.task011-identity-hotfix-',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as out:
            out.write(corrected)
            out.flush()
            os.fsync(out.fileno())
        # Guard against uncoordinated source change in between SHA check and publish.
        if sha(p.read_bytes())!=ORIGINAL_SHA:
            raise ValueError('TASK011_SOURCE_CHANGED_DURING_HOTFIX_STOP')
        os.replace(tmp,p)
    finally:
        Path(tmp).unlink(missing_ok=True)
    if sha(p.read_bytes())!=FIXED_SHA:
        raise ValueError('HOTFIX_POST_WRITE_SHA_MISMATCH')
    return {'status':'TASK011_IDENTITY_SCHEMA_HOTFIX_PASS',
            'old_sha256':ORIGINAL_SHA,'sha256':FIXED_SHA,
            'modified_file':str(p.relative_to(root)),
            'final_test_read':False,'checkpoints_modified':False,
            'accepted_artifacts_modified':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('.'))
    opts=ap.parse_args()
    print(json.dumps(fix(opts.root),sort_keys=True))
