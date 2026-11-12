"""TASK-008E BTC Validation runner. Restricted to 2023-2024, never any Final Test."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from cita2027.btc.validation import inputs,worker,evaluate,check_manifest,MODELS,VAL_SHA,VAL_COUNT


def run(root:Path):
    root=root.resolve()
    env={**os.environ,'PYTHONPATH':os.pathsep.join([str(root/'src'),str(root/'scripts')])}
    subprocess.run([sys.executable,'-m','pytest','-q',str(root/'tests'/'test_task008e_btc_validation.py')],cwd=root,env=env,check=True)
    df,plan,ident,_,_,_,_=inputs(root)
    print(json.dumps({'status':'BTC_VALIDATION_PREFLIGHT_PASS','plan_sha256':VAL_SHA,
         'count_per_model':len(plan['origins']),'first_origin':plan['origins'][0]['origin_date'],
         'first_target':plan['origins'][0]['target_date'],'final_test_read':False}),flush=True)
    # Separate per-model venv/GPU process, sequentially; each worker independently rechecks hashes.
    for model in MODELS:
        interpreter=root/('.venv' if model=='chronos' else '.venv-moirai')/'Scripts'/'python.exe'
        if not interpreter.is_file():raise ValueError('LOCKED_VENV_MISSING:'+model)
        cmd=[str(interpreter),str(root/'scripts'/'task008e_btc_validation.py'),
             '_worker','--root',str(root),'--model',model,'--allow-btc-validation']
        subprocess.run(cmd,cwd=root,env=env,check=True)
    for model in MODELS:
        check_manifest(root,model,plan,ident[model],df)
    return evaluate(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','_worker'])
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--allow-btc-validation',action='store_true')
    a=parser.parse_args()
    if not a.allow_btc_validation:raise ValueError('EXPLICIT_VALIDATION_SCOPE_AUTHORIZATION_REQUIRED')
    root=a.root.resolve()
    if a.command=='run':
        if a.model is not None:raise ValueError('UNEXPECTED_MODEL_ARGUMENT')
        output=run(root)
    else:
        if a.model is None:raise ValueError('WORKER_MODEL_REQUIRED')
        output=worker(root,a.model)
    print(json.dumps(output,indent=2,sort_keys=True,allow_nan=False),flush=True)

if __name__=='__main__':main()
