#!/usr/bin/env python3
"""Build both project instructions against the installed Spike headers (macOS)."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parent

def main():
    if sys.platform != 'darwin':
        raise SystemExit('The TFLM workflow currently supports macOS arm64; see README prerequisites.')
    spike=shutil.which('spike')
    if not spike: raise SystemExit('Install Spike with its matching development headers first.')
    prefix=Path(os.environ.get('SPIKE_PREFIX',str(Path(spike).resolve().parent.parent)))
    if not (prefix/'include/riscv/extension.h').exists():
        raise SystemExit('Missing matching Spike headers; set SPIKE_PREFIX.')
    (ROOT/'build').mkdir(exist_ok=True)
    subprocess.run(['clang++','-std=c++17','-O2','-fPIC','-dynamiclib','-undefined','dynamic_lookup',
        '-I'+str(prefix/'include'),str(ROOT.parent/'qmac8-requant-spike/extension.cc'),
        '-o',str(ROOT/'build/libqmac8.dylib')],check=True)
if __name__=='__main__': main()
