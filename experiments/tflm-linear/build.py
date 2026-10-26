#!/usr/bin/env python3
"""Build a minimal upstream TFLM library, then the operator test application."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
VENDOR=ROOT/'vendor/tflite-micro'
BUILD=ROOT/'build'
REVISION='8f6596d99898deaac36ebd8ff8bf1b9754fd2cd1'
COMMON=['-std=c++17','-O2','-fno-rtti','-fno-exceptions','-fno-threadsafe-statics',
        '-ffunction-sections','-fdata-sections','-fno-unwind-tables','-fno-asynchronous-unwind-tables',
        '-DTFLM_REMOVE_VIRTUAL_DELETE','-DTF_LITE_STATIC_MEMORY','-DTF_LITE_DISABLE_X86_NEON',
        '-DTFLITE_SINGLE_ROUNDING=1']
INCLUDES=['-I'+str(VENDOR),'-I'+str(VENDOR/'tensorflow/lite/micro/tools/make/downloads/flatbuffers/include'),
          '-I'+str(VENDOR/'tensorflow/lite/micro/tools/make/downloads'),
          '-I'+str(VENDOR/'tensorflow/lite/micro/tools/make/downloads/kissfft')]


def run(cmd, log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as stream:
        stream.write('COMMAND '+json.dumps([str(x) for x in cmd])+'\n'); stream.flush()
        proc=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f'{log}:\n'+log.read_text()[-5000:])


def main():
    args=argparse.ArgumentParser()
    args.add_argument('--library-only',action='store_true')
    args.add_argument('--target',choices=['host','riscv'],default='host')
    args.add_argument('--custom',action='store_true')
    args.add_argument('--qmac-only',action='store_true')
    args.add_argument('--gelu',action='store_true')
    args.add_argument('--shape',choices=['2x8','197x768'],default='197x768')
    args.add_argument('--project-kernel',action='store_true')
    args.add_argument('--block',type=int,choices=[1,2,4],default=1)
    args.add_argument('--fused',action='store_true')
    a=args.parse_args()
    if a.fused: a.gelu=True
    if a.block>1 or a.fused: a.custom=True
    if a.qmac_only: a.custom=True
    shape=a.shape+('-gelu' if a.gelu else '')+('-fused' if a.fused else '')
    assert subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip()==REVISION
    if a.target=='host':
        compiler='clang++'; ar='ar'; flags=COMMON
    else:
        prefix=ROOT/'vendor/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-'
        compiler=str(prefix)+'g++'; ar=str(prefix)+'ar'
        flags=COMMON+['-march=rv64imc','-mabi=lp64','-mcmodel=medany','-fno-use-cxa-atexit']
    libdir=BUILD/a.target
    libdir.mkdir(parents=True,exist_ok=True)
    all_sources=subprocess.check_output([str(ROOT/'vendor/make'),'-s','-f',
        'tensorflow/lite/micro/tools/make/Makefile','list_library_sources','SKIP_DOWNLOADS=1'],
        cwd=VENDOR,text=True).split()
    selected=[]
    for src in all_sources:
        if not src.startswith('tensorflow/lite/micro/'):
            continue
        if any(token in src for token in ['test_helper','test_helpers','mock_micro','fake_micro','debug_log.cc']):
            continue
        if '/kernels/' in src and '/kernels/internal/' not in src and Path(src).name not in [
            'fully_connected.cc','fully_connected_common.cc','kernel_util.cc']:
            continue
        selected.append(src)
    # Source selection follows the upstream list, restricted to one operator.
    objects=[]
    def compile_source(src):
        obj=libdir/'obj'/src.replace('/','_').replace('.cc','.o')
        objects.append(obj)
        command=[compiler]+flags+INCLUDES+['-c',str(VENDOR/src),'-o',str(obj)]
        signature=hashlib.sha256((json.dumps(command)+hashlib.sha256((VENDOR/src).read_bytes()).hexdigest()).encode()).hexdigest()
        stamp=obj.with_suffix('.stamp')
        if obj.exists() and stamp.exists() and stamp.read_text()==signature: return
        obj.parent.mkdir(parents=True,exist_ok=True)
        run(command,obj.with_suffix('.log'))
        stamp.write_text(signature)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(compile_source,selected))
    library=libdir/'libtflm.a'
    run([ar,'rcs',str(library)]+[str(o) for o in sorted(objects)],libdir/'archive.log')
    if a.library_only:
        print(library)
        return
    variant='qmac-only' if a.qmac_only else ('custom' if a.custom else 'baseline')
    if a.project_kernel: variant+='-kernel'
    if a.block>1: variant+=f'-b{a.block}'
    out=BUILD/shape/(a.target+'-'+variant)
    out.mkdir(parents=True,exist_ok=True)
    appobjects=[]
    for src in ['main.cc','fc_adapter.cc','platform.cc']+(['gelu_lut.cc'] if a.gelu else []):
        obj=out/(src+'.o')
        command=[compiler]+flags+INCLUDES+['-I'+str(BUILD/shape)]
        if a.custom: command+=['-DUSE_CUSTOM_FC']
        if a.qmac_only: command+=['-DUSE_QMAC_ONLY']
        if a.gelu: command+=['-DUSE_GELU_LUT']
        if a.fused: command+=['-DUSE_FUSION']
        if a.project_kernel: command+=['-DUSE_PROJECT_KERNEL']
        command += [f'-DPROJECT_BLOCK={a.block}']
        run(command+['-c',str(ROOT/src),'-o',str(obj)],out/(src+'.log'))
        appobjects.append(str(obj))
    exe=out/('program.elf' if a.target=='riscv' else 'program')
    command=[compiler]+flags+appobjects+[str(library)]
    if a.target=='riscv':
        start=out/'start.o'
        run([compiler]+flags+['-c',str(ROOT/'start.S'),'-o',str(start)],out/'start.log')
        command += [str(start),'-nostartfiles','-Wl,--gc-sections,--no-relax,-Map='+str(out/'program.map'),
                    '-T',str(ROOT/'link.ld'),'-lm','-lc','-lgcc']
    else:
        command += ['-Wl,-dead_strip']
    run(command+['-o',str(exe)],out/'link.log')
    if a.target=='host': run([str(exe)],out/'result.txt')
    print(exe)
    print((out/'result.txt').read_text() if a.target=='host' else 'Built RV64IMC/lp64 ELF')
    (out/'build.json').write_text(json.dumps(dict(tflm_revision=REVISION,
        compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
        flags=flags,includes=INCLUDES,sources=selected,
        elf_sha256=hashlib.sha256(exe.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__': main()
