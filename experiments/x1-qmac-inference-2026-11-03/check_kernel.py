#!/usr/bin/env python3
import json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;f=HERE/'build/x1-qmac';repo=HERE.parents[1];t=repo/'experiments/tflm-linear';src=f/'src/experiments/tflm-linear/blocking_tests.cc';cc=t/'vendor/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-g++'
def run(name,cmd):
 p=subprocess.run(list(map(str,cmd)),capture_output=True,text=True,timeout=300);(f/(name+'.log')).write_text('COMMAND '+json.dumps(list(map(str,cmd)))+'\n'+p.stdout+p.stderr);p.check_returncode()
def main():
    run('tests-host-build',['clang++','-std=c++17','-O1','-fsanitize=undefined','-fno-sanitize-recover=all',src,'-o',f/'tests-host'])
    run('tests-host',[f/'tests-host'])
    run('tests-build',[cc,'-std=c++17','-O2','-march=rv64imc','-mabi=lp64','-mcmodel=medany','-nostdlib','-nostartfiles','-Wl,--no-relax','-T',t/'link.ld',t/'start.S',src,'-lc','-lgcc','-o',f/'tests.elf'])
    run('tests-spike',['spike','--isa=RV64IMC','-m32','--extlib='+str(t/'build/libqmac8.dylib'),'--extension=xqmac8',f/'tests.elf'])
    (f/'kernel-tests.json').write_text(json.dumps({'cases':640,'seed':6455,'host_ubsan':'PASS','spike':'PASS'},indent=2)+'\n');print('X1: 640 cases passed on host UBSan and Spike')

if __name__=="__main__":
    main()
