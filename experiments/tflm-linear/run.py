#!/usr/bin/env python3
"""Offline rebuild, correctness checks and full/tile comparison after bootstrap."""
import hashlib
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
BUILD=ROOT/'build'


def run(command):
    print('+ '+' '.join(str(v) for v in command),flush=True)
    subprocess.run(command,cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gelu',action='store_true')
    args=parser.parse_args()
    graph_options=['--gelu'] if args.gelu else []
    suffix='-gelu' if args.gelu else ''
    BUILD.mkdir(exist_ok=True)
    py=Path(os.environ.get('MODEL_PYTHON',ROOT.parent/'deit-tiny-inspect/.venv/bin/python')).absolute()
    reference=ROOT.parent/'deit-linear-int8/build/operator-reference.npz'
    if not reference.exists(): run([str(py),str(ROOT.parent/'deit-linear-int8/prepare.py')])
    run([str(py),'export_model.py']+graph_options)
    # Rebuild the same extension source previously verified with ISA tests.
    plugin=BUILD/'libqmac8.dylib'
    run([sys.executable,'build_extension.py'])
    for shape in ['2x8','197x768']:
        run([sys.executable,'build.py','--target','host','--shape',shape]+graph_options)
        for options in [[],['--qmac-only'],['--custom']]:
            run([sys.executable,'build.py','--target','riscv','--shape',shape]+options+graph_options)
            run([sys.executable,'profile.py','--shape',shape]+options+graph_options)
        # Additional stock-Spike pass, independent of the extension loader.
        run(['spike','--isa=RV64IMC','-m32',str(BUILD/(shape+suffix)/'riscv-baseline/program.elf')])
    comparison={}
    for shape in ['2x8','197x768']:
        shape+=suffix
        baseline=json.loads((BUILD/shape/'riscv-baseline/profile.json').read_text())
        custom=json.loads((BUILD/shape/'riscv-custom/profile.json').read_text())
        qmac_only=json.loads((BUILD/shape/'riscv-qmac-only/profile.json').read_text())
        assert baseline['model_sha256']==custom['model_sha256']
        assert qmac_only['model_sha256']==custom['model_sha256']
        comparison[shape]=dict(baseline=baseline,qmac_only=qmac_only,custom=custom,
            instruction_count_ratio=baseline['invoke_retired']/custom['invoke_retired'],
            instruction_reduction_fraction=1-custom['invoke_retired']/baseline['invoke_retired'],
            requant_incremental_reduction_fraction=1-custom['invoke_retired']/qmac_only['invoke_retired'])
    (BUILD/('comparison'+suffix+'.json')).write_text(json.dumps(comparison,indent=2)+'\n')
    local=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.cc','.S','.ld','.md']]
    local += [ROOT.parent/'qmac8-requant-spike'/n for n in ['extension.cc','qmac8.h','requant8.h']]
    local += [reference,ROOT.parent/'deit-tiny-inspect/results/model-manifest.json',plugin]
    for shape in ['2x8','197x768']:
        shape+=suffix
        local += [BUILD/shape/n for n in ['model.tflite','model_data.h','reference.json']]
    for name in ['flatbuffers','kissfft','eyalroz_printf']:
        local.append(ROOT/'vendor/tflite-micro/tensorflow/lite/micro/tools/make/downloads'/name/'.download_complete')
    hashes={str(p.relative_to(ROOT.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in local}
    (BUILD/('manifest'+suffix+'.json')).write_text(json.dumps(dict(
        tflm_revision=subprocess.check_output(['git','-C',str(ROOT/'vendor/tflite-micro'),'rev-parse','HEAD'],text=True).strip(),
        tflm_worktree_status=subprocess.check_output(['git','-C',str(ROOT/'vendor/tflite-micro'),'status','--porcelain','--untracked-files=no'],text=True),
        python_packages=subprocess.check_output([str(py),'-m','pip','freeze'],text=True).splitlines(),
        spike_version=subprocess.run(['spike','--help'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True).stdout.splitlines()[0],
        sha256=hashes),indent=2)+'\n')
    print('PASS: TFLM full fc1'+(' -> GELU LUT' if args.gelu else '')+
          ' and tile, host/stock-Spike/custom-ISA, every INT8 output checked.')


if __name__=='__main__': main()
