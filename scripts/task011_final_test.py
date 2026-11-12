"""TASK-011 single-CLI Final Test lock -> frozen inference -> causal evaluation -> audit.

Important: --allow-final-test explicitly authorizes irreversible use of holdout.
If Phase A has any blocker, Phase B is NEVER attempted. Resume is write-once.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from cita2027.finaltest import lock as L
from cita2027.finaltest import execute as E


def python_venv(root:Path,model:str)->Path:
    p=root/('.venv' if model=='chronos' else '.venv-moirai')/'Scripts'/'python.exe'
    if not p.is_file():raise ValueError('FROZEN_MODEL_PYTHON_VENV_MISSING:'+model)
    return p


def probe(root:Path,model:str):
    """Only local model weight SHA; no CUDA allocation or Test access."""
    import yaml
    from pilot_forecast_cache import identities,verify_local_checkpoint
    conf=yaml.safe_load((root/'configs'/'smoke.yaml').read_text('utf-8'))
    audit=json.loads((root/'artifacts'/'checkpoints'/'hf_audit.json').read_text('utf-8'))
    _,id_,rev,files=identities(conf,audit,model)
    weight=verify_local_checkpoint(id_,rev,files)
    got={'id':id_,'revision':rev,'weight_sha256':weight}
    if got!=L.CHECKPOINTS[model]:raise ValueError('CHECKPOINT_SHA_NOT_ACCEPTED:'+model)
    return got


def sub(root:Path,cmd:list[str],interpreter:Path):
    env={**os.environ,'PYTHONPATH':os.pathsep.join([str(root/'src'),str(root/'scripts')])}
    if cmd and cmd[0]=='_worker':
        # Stream actual GPU progress to Windows console; do not buffer hours of logs.
        subprocess.run([str(interpreter),str(root/'scripts'/'task011_final_test.py'),*cmd],
                       cwd=root,env=env,check=True)
        return {'status':'WORKER_EXITED_ZERO'}
    p=subprocess.run([str(interpreter),str(root/'scripts'/'task011_final_test.py'),*cmd],
                     cwd=root,env=env,check=True,capture_output=True,text=True)
    if p.stderr:print(p.stderr,file=sys.stderr,flush=True)
    try:out=json.loads(p.stdout)
    except json.JSONDecodeError as exc:
        print(p.stdout,file=sys.stderr);raise ValueError('WORKER_DID_NOT_RETURN_VALID_JSON') from exc
    print(json.dumps(out,sort_keys=True),flush=True)
    return out


def run(root:Path):
    root=root.resolve()
    env={**os.environ,'PYTHONPATH':os.pathsep.join([str(root/'src'),str(root/'scripts')])}
    # Only newly added focused tests; never rerun accepted Dev/Validation suites.
    subprocess.run([sys.executable,'-m','pytest','-q',str(root/'tests'/'test_task011_final_test.py')],
                   cwd=root,env=env,check=True)
    # Hard Phase A identity + checkpoint weight checks in each native venv.
    checks={}
    for model in L.MODELS:
        x=sub(root,['_checkpoint_probe','--root',str(root),'--model',model],python_venv(root,model))
        if x['status']!='LOCAL_CHECKPOINT_SHA_PASS':raise ValueError('CHECKPOINT_PROBE_FAILED')
        checks[model]=x['checkpoint']
    frozen=L.preflight_and_lock(root,checks)
    print(json.dumps(frozen,sort_keys=True,indent=2),flush=True)
    # Only after durable SHA-checked lock: Phase B loads any realized target data.
    anchor=frozen['pretest_manifest_sha256']
    L.load_lock(root,anchor)
    print(json.dumps({'gate':'PHASE_B_AUTHORIZED','pretest_sha256':anchor,
                      'conditions':'LOCK_VERIFIED_BEFORE_ANY_TARGET_DECODING'}),flush=True)
    # Sequential, GPU memory released in each interpreter between workers.
    for market in ('spy','btcusdt'):
        for model in L.MODELS:
            sub(root,['_worker','--root',str(root),'--market',market,
                      '--model',model,'--lock-sha',anchor],python_venv(root,model))
    summary=E.evaluate(root,anchor)
    print(json.dumps({'gate':'FINAL_CALIBRATION_COMPLETE','result':summary},sort_keys=True),flush=True)
    manuscript=E.manuscript_update(root,anchor)
    audit=E.independent_audit_and_freeze(root,anchor)
    return {'status':'TASK011_FINAL_TECHNICAL_AUDIT_PASS_NOT_SCIENTIFIC_CLEARANCE',
            'phase_a':frozen,'evaluation':summary,'audit':audit,'manuscript':manuscript,
            'scientific_clearance':False,'contamination_free_verified':False,
            'point_in_time_adjusted_prices_verified':False,
            'btc_historical_point_in_time_verified':False}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['run','_worker','_checkpoint_probe'])
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--market',choices=['spy','btcusdt'])
    ap.add_argument('--model',choices=L.MODELS)
    ap.add_argument('--lock-sha')
    ap.add_argument('--allow-final-test',action='store_true')
    a=ap.parse_args();root=a.root.resolve()
    if a.command=='run':
        if not a.allow_final_test:raise ValueError('EXPLICIT_FINAL_TEST_AUTHORIZATION_REQUIRED')
        obj=run(root)
    elif a.command=='_worker':
        if not a.market or not a.model or not a.lock_sha:
            raise ValueError('INCOMPLETE_INTERNAL_WORKER_ARGUMENTS')
        # Internal work is scoped to an already durable pretest SHA. No CLI can
        # construct a plan by reading target values.
        L.load_lock(root,a.lock_sha)
        obj=E.worker(root,a.market,a.model,a.lock_sha)
    else:
        if not a.model:raise ValueError('MODEL_REQUIRED_FOR_WEIGHT_SHA_PROBE')
        obj={'status':'LOCAL_CHECKPOINT_SHA_PASS','model':a.model,
             'checkpoint':probe(root,a.model),'gpu_used':False,'final_test_read':False}
    print(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False),flush=True)

if __name__=='__main__':main()
