#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
def run(cmd,log):
    with (ROOT/'build'/log).open('w') as f:
        p=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode: raise RuntimeError((ROOT/'build'/log).read_text()[-5000:])
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tests-only',action='store_true')
    args=parser.parse_args()
    (ROOT/'build').mkdir(exist_ok=True)
    run([sys.executable,'build_extension.py'],'extension-build.log')
    py=ROOT.parent/'deit-tiny-inspect/.venv/bin/python'
    run([str(py),'export_model.py','--gelu'],'export-gelu-opt.log')
    run([str(py),'export_model.py','--fused'],'export-fused.log')
    # Exhaustive dimension/tail/alignment combinations with independent scalar reference.
    run(['clang++','-std=c++17','-O1','-fsanitize=undefined','-fno-sanitize-recover=all',
         'blocking_tests.cc','-o','build/blocking-tests-host'],'blocking-tests-host-build.log')
    run(['build/blocking-tests-host'],'blocking-tests-host.log')
    cc=ROOT/'vendor/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-g++'
    run([str(cc),'-std=c++17','-O2','-march=rv64imc','-mabi=lp64','-mcmodel=medany',
         '-nostdlib','-nostartfiles','-Wl,--no-relax',
         '-T','link.ld','start.S','blocking_tests.cc','-lc','-lgcc','-o','build/blocking-tests.elf'],'blocking-tests-build.log')
    run(['spike','--isa=RV64IMC','-m32','--extlib='+str(ROOT/'build/libqmac8.dylib'),
         '--extension=xqmac8','build/blocking-tests.elf'],'blocking-tests-spike.log')
    print('PASS: 640 blocking/fusion tail/alignment cases, host UBSan and Spike.',flush=True)
    if args.tests_only: return
    results=[]
    for shape in ['2x8','197x768']:
        for fused in [False,True]:
            for block in [1,2,4]:
                options=['--gelu','--custom','--shape',shape,'--block',str(block)]+(['--fused'] if fused else [])+(['--project-kernel'] if block==1 else [])
                run([sys.executable,'build.py','--target','host']+options,f'opt-host-{shape}-{block}-{fused}.log')
                run([sys.executable,'build.py','--target','riscv']+options,f'opt-build-{shape}-{block}-{fused}.log')
                run([sys.executable,'profile.py']+options,f'opt-profile-{shape}-{block}-{fused}.log')
                folder=ROOT/'build'/(shape+'-gelu'+('-fused' if fused else ''))/('riscv-custom'+('-kernel' if block==1 else f'-b{block}'))
                p=json.loads((folder/'profile.json').read_text());results.append(p)
                print(shape,'block',block,'fused',fused,p['invoke_retired'],flush=True)
    (ROOT/'build/optimizations.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__': main()
