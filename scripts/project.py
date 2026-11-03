#!/usr/bin/env python3
"""Repository entry point. Run from any directory; outputs stay inside this repo."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'experiments'
PY=E/'deit-tiny-inspect/.venv/bin/python'
LOG=ROOT/'build/logs'
ENV=dict(os.environ,MODEL_PYTHON=str(PY),PYTHONUNBUFFERED='1')

def run(folder, *command):
    LOG.mkdir(parents=True,exist_ok=True)
    options=''.join('-'+str(arg).lstrip('-') for arg in command[2:] if str(arg).startswith('--'))
    log=LOG/(folder+'-'+Path(command[1] if len(command)>1 else command[0]).stem+options+'.log')
    print('+',folder, ' '.join(map(str,command)), '\n  log:',log.relative_to(ROOT),flush=True)
    with log.open('w') as f:
        p=subprocess.run(list(map(str,command)),cwd=E/folder,env=ENV,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:
        print(log.read_text()[-6000:],file=sys.stderr)
        raise SystemExit(f'FAILED ({p.returncode}); see {log}')
    print('  PASS',flush=True)

def setup():
    if not PY.exists():
        subprocess.run([sys.executable,'-m','venv',str(PY.parent.parent)],check=True)
    run('deit-tiny-inspect',PY,'-m','pip','install','-r','requirements-lock.txt')
    run('tflm-linear',sys.executable,'bootstrap.py')
    run('deit-tiny-inspect',PY,'download_weights.py')

def fixtures():
    if not PY.exists(): raise SystemExit('Run: python3 scripts/project.py setup')
    manifest=E/'deit-tiny-inspect/results/model-manifest.json'
    if not manifest.exists():
        raise SystemExit('Missing checkpoint manifest. Run setup first (downloads official weights).')
    run('deit-tiny-inspect',PY,'inspect_model.py')
    run('deit-linear-int8',PY,'prepare.py')

def test():
    fixtures()
    run('qmac8-requant-spike',sys.executable,'run.py')
    run('tflm-linear',sys.executable,'benchmark_optimizations.py','--tests-only')

def bench():
    fixtures()
    run('tflm-linear',sys.executable,'run.py')
    run('tflm-linear',sys.executable,'run.py','--gelu')
    run('tflm-linear',sys.executable,'compare_stages.py')
    run('tflm-linear',sys.executable,'benchmark_optimizations.py')
    run('tflm-linear',sys.executable,'report_optimizations.py')

def full():
    fixtures()
    run('deit-full-int8',sys.executable,'run.py')

def tpu():
    run('tpu-spike',sys.executable,'run.py')

def checkpoint():
    # Each step writes its own ignored build folder; the collector uses fresh
    # B0, B1, X1, X2 and TPU measurements, not committed reference numbers.
    test()
    full()
    run('tpu-spike',sys.executable,'build_extension.py')
    run('b0-full-inference-2026-11-03',sys.executable,'run.py')
    run('full-inference-baseline-2026-11-03',sys.executable,'run.py')
    run('x1-qmac-inference-2026-11-03',sys.executable,'run.py')
    run('x1-qmac-inference-2026-11-03',sys.executable,'check_kernel.py')
    run('tpu-evaluation-2026-11-02',sys.executable,'evaluate.py','--variants','qmac-inline','tpu-inline-hoist')
    run('b0-full-inference-2026-11-03',sys.executable,'collect.py')
    print('Five-stage summary: experiments/b0-full-inference-2026-11-03/build/REPORT.md',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['setup','fixtures','test','bench','full','tpu','checkpoint','all'])
    a=parser.parse_args()
    if a.stage=='all':
        test();bench();full()
    else: globals()[a.stage]()
    print('Completed:',a.stage,flush=True)
if __name__=='__main__': main()
