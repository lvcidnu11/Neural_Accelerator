#!/usr/bin/env python3
"""Reproduce full synthetic-input inference and save evidence manifest."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
B=ROOT/'build'
def main():
    B.mkdir(exist_ok=True)
    py=Path(os.environ.get('MODEL_PYTHON',ROOT.parent/'deit-tiny-inspect/.venv/bin/python')).absolute()
    commands=[[sys.executable,str(ROOT.parent/'tflm-linear/build_extension.py')],
              [sys.executable,str(ROOT.parent/'tflm-linear/build.py'),'--target','host','--library-only'],
              [sys.executable,str(ROOT.parent/'tflm-linear/build.py'),'--target','riscv','--library-only'],
              [str(py),'export_model.py'],[sys.executable,'build.py','--target','host'],
              [sys.executable,'build.py','--target','host','--sanitize'],
              [sys.executable,'build.py','--target','riscv'],[sys.executable,'profile_spike.py']]
    for i,command in enumerate(commands):
        print('+ '+' '.join(command),flush=True)
        with (B/f'run-step-{i}.log').open('w') as f:
            p=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError((B/f'run-step-{i}.log').read_text()[-4000:])
    record_manifest()
    print('PASS: complete 12-block DeiT-Tiny integer inference, host/UBSan/RV64IMC Spike; 1000 logits and 14 stage hashes.')
def record_manifest():
    files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.cc','.py','.md']]
    files += [ROOT.parent/'tflm-linear'/n for n in ['linear_kernel.h','start.S','link.ld','platform.cc']]
    files += [B/n for n in ['model.tflite','input.bin','expected.bin','reference.json','reference.h','model.S','profile.json']]
    for variant in ['host','host-ubsan','riscv']:
        files += [B/variant/n for n in ['build.json','program.elf' if variant=='riscv' else 'program']]
    files += [ROOT.parent/'tflm-linear/build/libqmac8.dylib',ROOT.parent/'qmac8-requant-spike/extension.cc',
              ROOT.parent/'deit-tiny-inspect/results/model-manifest.json']
    hashes={str(p.relative_to(ROOT.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (B/'manifest.json').write_text(json.dumps(dict(sha256=hashes,
        tflm_revision='8f6596d99898deaac36ebd8ff8bf1b9754fd2cd1',
        reproducibility='See each build.json and saved Spike debugger/histogram command JSON files'),indent=2)+'\n')
if __name__=='__main__':main()
