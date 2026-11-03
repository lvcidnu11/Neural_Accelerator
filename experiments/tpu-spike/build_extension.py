#!/usr/bin/env python3
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    prefix=Path(os.environ.get('SPIKE_PREFIX',str(Path(shutil.which('spike')).resolve().parent.parent)))
    (ROOT/'build').mkdir(exist_ok=True)
    suffix='dylib' if sys.platform=='darwin' else 'so'
    flags=['-dynamiclib','-undefined','dynamic_lookup'] if sys.platform=='darwin' else ['-shared']
    subprocess.run(['c++','-std=c++20','-O2','-Wall','-Wextra','-Werror','-fPIC']+flags+
        ['-I'+str(prefix/'include'),str(ROOT/'extension.cc'),'-o',str(ROOT/'build'/('libtpu.'+suffix))],check=True)
