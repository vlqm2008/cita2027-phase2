"""TASK-008A single entry point: tests + bounded archive ingestion; no GPU."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--skip-focused-tests', action='store_true', help='Only for CI that already executed the tests')
    args=p.parse_args()
    root=args.root.resolve()
    if not args.skip_focused_tests:
        env={**os.environ, 'PYTHONPATH':os.pathsep.join([str(root/'src'),str(root/'scripts')])}
        subprocess.run([sys.executable,'-m','pytest','-q',str(root/'tests'/'test_task008a_btc_ingest.py')],cwd=root,env=env,check=True)
    from cita2027.btc.ingest import build
    result=build(root)
    print(json.dumps(result,indent=2,sort_keys=True),flush=True)

if __name__ == '__main__':main()
