#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'tflm-linear'
BUILD=ROOT/'build'
spec=importlib.util.spec_from_file_location('tflm_build',PARENT/'build.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target',choices=['host','riscv'],default='host')
    parser.add_argument('--sanitize',action='store_true')
    a=parser.parse_args();out=BUILD/(a.target+('-ubsan' if a.sanitize else ''));out.mkdir(parents=True,exist_ok=True)
    if a.target=='host': compiler='clang++';flags=base.COMMON+(['-fsanitize=undefined','-fno-sanitize-recover=all'] if a.sanitize else [])
    else:
        compiler=str(PARENT/'vendor/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-g++')
        flags=base.COMMON+['-march=rv64imc','-mabi=lp64','-mcmodel=medany','-fno-use-cxa-atexit']
    library=out/'libtflm.a';shutil.copy2(PARENT/'build'/a.target/'libtflm.a',library)
    objects=[]
    for source in [ROOT/'main.cc',ROOT/'model.cc',PARENT/'platform.cc']:
        obj=out/(source.name+'.o')
        command=[compiler]+flags+base.INCLUDES+['-I'+str(BUILD),'-DMODEL_FILE="'+str(BUILD/'model.tflite')+'"',
            '-DINPUT_FILE="'+str(BUILD/'input.bin')+'"','-c',str(source),'-o',str(obj)]
        base.run(command,out/(source.name+'.log'));objects.append(str(obj))
    exe=out/('program.elf' if a.target=='riscv' else 'program')
    command=[compiler]+flags+objects+[str(library)]
    if a.target=='riscv':
        for source in [PARENT/'start.S',BUILD/'model.S']:
            obj=out/(source.name+'.o');base.run([compiler]+flags+['-c',str(source),'-o',str(obj)],out/(source.name+'.log'))
            command.append(str(obj))
        command+=['-nostartfiles','-Wl,--gc-sections,--no-relax,-Map='+str(out/'program.map'),'-T',str(PARENT/'link.ld'),'-lm','-lc','-lgcc']
    else: command+=['-Wl,-dead_strip']
    base.run(command+['-o',str(exe)],out/'link.log')
    if a.target=='host': base.run([str(exe)],out/'result.txt');print((out/'result.txt').read_text())
    (out/'build.json').write_text(json.dumps(dict(compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
        flags=flags,elf_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest()),indent=2)+'\n')
    print(exe)
if __name__=='__main__':main()
