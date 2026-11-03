#!/usr/bin/env python3
"""Test the tile ISA and reproduce controlled full-inference comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1];B=ROOT/'build';F=ROOT.parent/'deit-full-int8';T=ROOT.parent/'tflm-linear'
COMMANDS=[]
def run(cmd,name):
    cmd=list(map(str,cmd));COMMANDS.append(dict(argv=cmd,cwd=str(ROOT)))
    p=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=900)
    (B/(name+'.log')).write_text(p.stdout)
    if p.returncode:raise RuntimeError(name+'\n'+p.stdout[-5000:])
    return p.stdout

def tests():
    run([sys.executable,ROOT/'build_extension.py'],'build-extension')
    run([sys.executable,T/'build_extension.py'],'build-qmac')
    cc=T/'vendor/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-g++'
    flags=['-std=c++17','-O2','-march=rv64imc','-mabi=lp64','-mcmodel=medany','-nostdlib','-nostartfiles',
           '-Wl,--no-relax','-T',T/'link.ld',T/'start.S']
    ext=['--extlib='+str(T/'build/libqmac8.dylib'),'--extension=xqmac8',
         '--extlib='+str(B/'libtpu.dylib'),'--extension=xtpu']
    run(['clang++','-std=c++17','-O1','-fsanitize=undefined','-fno-sanitize-recover=all',ROOT/'tests.cc','-o',B/'tests-host'],'host-build')
    run([B/'tests-host'],'host-tests')
    run([cc]+flags+[ROOT/'tests.cc','-lc','-lgcc','-o',B/'tests.elf'],'target-build')
    run([cc]+flags+['-DLOG_SMOKE',ROOT/'tests.cc','-lc','-lgcc','-o',B/'tests-logged.elf'],'target-logged-build')
    for mode in ['fast','logged']:
        extra=['-l','--log-commits','--log='+str(B/'commits.log')] if mode=='logged' else []
        run(['spike','--isa=RV64IMC','-m32']+ext+extra+[B/('tests-logged.elf' if mode=='logged' else 'tests.elf')],'tests-'+mode)
    for case in range(9):
        exe=B/f'negative-{case}.elf';log=B/f'negative-{case}-trace.log'
        run([cc]+flags+[f'-DCASE={case}',ROOT/'negative.cc','-lc','-lgcc','-o',exe],f'negative-{case}-build')
        run(['spike','--isa=RV64IMC','-m32','--instructions=2000','-l','--log='+str(log)]+
            (ext if case else [])+[exe],f'negative-{case}')
        assert 'trap_illegal_instruction' in log.read_text(),case
    run(['spike','--help'],'spike-version')
    run([cc,'--version'],'compiler-version')
    (B/'verification.json').write_text(json.dumps(dict(kernel_cases=900,extrema_cases=1,logged_boundary_cases=4,negative_controls=9,host_ubsan='PASS',spike_fast='PASS',spike_logged='PASS',illegal_traps='PASS',seed=6455),indent=2)+'\n')
    print('PASS: 900 dimension/alignment/fusion cases + extrema, host UBSan and Spike fast, 4 logged boundary cases; 9 illegal-instruction controls.',flush=True)

def memory_bytes(profile):
    mn=profile['mnemonics'];read=write=0
    for name,count in mn.items():
        op=name.removeprefix('c.').removesuffix('sp')
        if op in ['lb','lbu','lh','lhu','lw','lwu','ld']:read+=count*dict(lb=1,lbu=1,lh=2,lhu=2,lw=4,lwu=4,ld=8)[op]
        if op in ['sb','sh','sw','sd']:write+=count*dict(sb=1,sh=2,sw=4,sd=8)[op]
    return dict(core_read_bytes=read,core_write_bytes=write,
        combined_read_bytes=read+profile.get('accelerator',{}).get('read_bytes',0),
        combined_write_bytes=write+profile.get('accelerator',{}).get('write_bytes',0))

def report():
    profiles={key:json.loads((F/'build'/file).read_text()) for key,file in [('baseline','profile.json'),('tpu','profile-tpu.json')]}
    assert profiles['baseline']['model_sha256']==profiles['tpu']['model_sha256']
    assert profiles['baseline']['correctness']==profiles['tpu']['correctness']
    for key,p in profiles.items():
        p.update(memory_bytes(p));p['macs_per_core_instruction']=p['mathematical_macs']/p['retired_instructions']
        p['core_memory_instructions_per_mac']=(p['mix']['load']+p['mix']['store'])/p['mathematical_macs']
        p['sections']={m[1]:int(m[2]) for line in (F/'build'/('riscv-tpu' if key=='tpu' else 'riscv')/'size.txt').read_text().splitlines()
                       if (m:=re.match(r'^(\.\S+)\s+(\d+)\s+',line))}
    old,new=[profiles[k]['retired_instructions'] for k in ['baseline','tpu']]
    out=REPO/'results/tpu';out.mkdir(parents=True,exist_ok=True)
    verification=json.loads((B/'verification.json').read_text());verification['full_model']=dict(nodes=14,blocks=12,exact_final_logits=1000,host_ubsan='PASS',spike='PASS')
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    for key,p in profiles.items():(out/(key+'.json')).write_text(json.dumps(p,indent=2)+'\n')
    summary=dict(scope='Same complete DeiT-Tiny synthetic-input inference and 14 hash checks; GCC 15.2.0 -O2; functional Spike',
        retired_reduction=1-new/old,instruction_count_ratio=old/new,
        modeled_engine_cycles=profiles['tpu']['accelerator']['serial_cycles_bw16'],
        limitations=['Instruction count ratio is not hardware speedup','Engine cycles assume 64 INT8 MAC PEs, k+m+n-2 schedule, serial DMA at 16 bytes/cycle, no overlap/cache/bank/conflict/setup cost',
                    'CPU remaining cycles and stalls are not modeled; no end-to-end cycle estimate','No RTL/area/Fmax/power measurement','Synthetic quantization; no accuracy improvement established'])
    (out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    paths=[p for folder in [ROOT,F,T] for p in folder.iterdir() if p.suffix in ['.cc','.h','.py','.S','.ld']]
    paths+=[B/'libtpu.dylib',T/'build/libqmac8.dylib',REPO/'scripts/project.py']
    paths+=[F/'build'/n for n in ['model.tflite','input.bin','expected.bin','reference.json']]
    paths+=[F/'build'/variant/n for variant in ['riscv','riscv-tpu','host-ubsan-tpu'] for n in ['build.json','program' if variant=='host-ubsan-tpu' else 'program.elf']]
    (out/'manifest.json').write_text(json.dumps(dict(sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        commands=[{'cwd':x['cwd'].replace(str(REPO),'<repo>'),'argv':[s.replace(str(REPO),'<repo>') for s in x['argv']]} for x in COMMANDS]),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--tests-only',action='store_true');p.add_argument('--report-only',action='store_true');a=p.parse_args()
    B.mkdir(exist_ok=True)
    if not a.report_only:
        tests()
        if a.tests_only:return
        for options in [[],['--tpu']]:
            label='tpu' if options else 'baseline'
            if options:run([sys.executable,F/'build.py','--target','host','--sanitize']+options,'full-host-ubsan')
            run([sys.executable,F/'build.py','--target','riscv']+options,'full-build-'+label)
            run([sys.executable,F/'profile_spike.py']+options,'full-profile-'+label)
    if COMMANDS:(B/'commands.json').write_text(json.dumps(COMMANDS,indent=2)+'\n')
    else: COMMANDS.extend(json.loads((B/'commands.json').read_text()))
    report()
if __name__=='__main__':main()
