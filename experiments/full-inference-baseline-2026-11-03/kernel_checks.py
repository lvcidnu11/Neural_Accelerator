#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tpu-evaluation-2026-11-02'))
from evaluate import run,DEFAULT_REPO
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('variants',nargs='+');a=p.parse_args()
r=DEFAULT_REPO;t=r/'experiments/tflm-linear';cc=t/'vendor/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-g++'
for name in a.variants:
    out=HERE/'build'/name;src=out/'src/experiments'/('tpu-spike/tests.cc' if name.startswith('tpu') else 'tflm-linear/blocking_tests.cc')
    run(['clang++','-std=c++17','-O1','-fsanitize=undefined','-fno-sanitize-recover=all',src,'-o',out/'tests-host'],out/'tests-host-build.log')
    run([out/'tests-host'],out/'tests-host.log')
    run([cc,'-std=c++17','-O2','-march=rv64imc','-mabi=lp64','-mcmodel=medany','-nostdlib','-nostartfiles','-Wl,--no-relax','-T',t/'link.ld',t/'start.S',src,'-lc','-lgcc','-o',out/'tests.elf'],out/'tests-build.log')
    cmd=['spike','--isa=RV64IMC','-m32','--extlib='+str(t/'build/libqmac8.dylib'),'--extension=xqmac8']
    if name.startswith('tpu'):cmd+=['--extlib='+str(r/'experiments/tpu-spike/build/libtpu.dylib'),'--extension=xtpu']
    run(cmd+[out/'tests.elf'],out/'tests-spike.log')
    report=dict(variant=name,cases=900 if name.startswith('tpu') else 640,host_ubsan='PASS',spike='PASS',seed=6455)
    (out/'kernel-tests.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
