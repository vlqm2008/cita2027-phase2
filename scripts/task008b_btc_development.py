"""TASK-008B: one command, two venv workers, small gate then full BTC Development.

Only BTC warm-up/Development is loaded; SPY accepted report is read-only for
within-market contrast descriptions. Never launches BTC Validation or any Final Test.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from cita2027.btc.development import (MODELS, DATA_SHA, DEV_SHA, DEV_COUNT, GATE_COUNT,
    load_inputs, run_worker, records, cross_model, cache_dir, validate_small_gate,
    verify_manifest, diagnostic_report)


def subprocess_worker(root: Path, model: str, limit: int):
    interpreter=root / ('.venv' if model=='chronos' else '.venv-moirai')/'Scripts'/'python.exe'
    if not interpreter.is_file():
        raise ValueError(f'MISSING_LOCKED_PYTHON_ENV_{model}: {interpreter}')
    cmd=[str(interpreter),str(root/'scripts'/'task008b_btc_development.py'),
         '_worker','--root',str(root),'--model',model,'--limit',str(limit),'--allow-btc-development']
    env={**os.environ,'PYTHONPATH':os.pathsep.join([str(root/'src'),str(root/'scripts')])}
    subprocess.run(cmd,cwd=root,env=env,check=True)


def run(root: Path) -> dict:
    root=root.resolve()
    env={**os.environ,'PYTHONPATH':os.pathsep.join([str(root/'src'),str(root/'scripts')])}
    tests=root/'tests'/'test_task008b_btc_development.py'
    subprocess.run([sys.executable,'-m','pytest','-q',str(tests)],cwd=root,env=env,check=True)
    df,plan,ident,_,_=load_inputs(root)
    # No GPU until full BTC dataset/plan provenance has passed and Gate A is invoked.
    print(json.dumps({'status':'BTC_DEV_PREFLIGHT_PASS','data_sha256':DATA_SHA,
        'plan_sha256':DEV_SHA,'count_per_model':len(plan['origins']),
        'small_batch_per_model':GATE_COUNT,'btc_validation_used':False}),flush=True)
    for model in MODELS:
        subprocess_worker(root,model,GATE_COUNT)
    gate=validate_small_gate(root)
    print(json.dumps(gate,sort_keys=True),flush=True)
    # Gate B is unreachable if one model fails Gate A.
    for model in MODELS:
        subprocess_worker(root,model,DEV_COUNT)
    # Recheck raw JSON records + model manifests before any CPU calibration.
    complete={}
    for model in MODELS:
        manifest,rr=verify_manifest(root,model,plan,ident[model],df)
        complete[model]=rr
    cross_model(complete['chronos'],complete['moirai'])
    result=diagnostic_report(root)
    result['small_batch_gate_status']=gate['status']
    result['forecast_manifests']={m:str(cache_dir(root,m)/'manifest.json') for m in MODELS}
    result['gate_for_btc_validation']='GO_TECHNICAL_SUBJECT_TO_REPORT_REVIEW_NO_AUTOMATIC_EXECUTION'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','_worker'))
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--limit',type=int,choices=(GATE_COUNT,DEV_COUNT))
    parser.add_argument('--allow-btc-development',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve()
    if args.command=='_worker':
        if args.model is None or args.limit is None or not args.allow_btc_development:
            # Internal subprocess authorization is passed explicitly by orchestrator.
            raise ValueError('BTC_DEV_WORKER_EXPLICIT_AUTHORIZATION_REQUIRED')
        result=run_worker(root,args.model,args.limit)
    else:
        if args.model is not None or args.limit is not None or not args.allow_btc_development:
            raise ValueError('BTC_DEV_FULL_RUN_EXPLICIT_AUTHORIZATION_REQUIRED')
        result=run(root)
    print(json.dumps(result,indent=2,allow_nan=False,sort_keys=True),flush=True)

if __name__=='__main__':main()
