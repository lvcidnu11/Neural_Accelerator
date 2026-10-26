#!/usr/bin/env python3
"""Download isolated, pinned dependencies; never install global toolchains."""
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
V=ROOT/'vendor'
REV='8f6596d99898deaac36ebd8ff8bf1b9754fd2cd1'


def call(command,cwd=None): subprocess.run(command,cwd=cwd,check=True)


def archive(name,url,expected):
    target=V/name
    if not target.exists(): call(['curl','-fL',url,'-o',str(target)])
    assert hashlib.sha256(target.read_bytes()).hexdigest()==expected,name
    return target


def main():
    if (platform.system(),platform.machine())!=('Darwin','arm64'):
        raise SystemExit('This bootstrap is pinned for macOS arm64; select an equivalent RV64IMC/lp64 newlib toolchain on other hosts.')
    V.mkdir(exist_ok=True)
    repo=V/'tflite-micro'
    if not (repo/'.git').exists():
        call(['git','init',str(repo)])
        call(['git','remote','add','origin','https://github.com/tensorflow/tflite-micro.git'],repo)
        call(['git','fetch','--depth','1','origin',REV],repo)
        call(['git','checkout','--detach',REV],repo)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==REV
    gnu=archive('make-4.4.1.tar.gz','https://ftp.gnu.org/gnu/make/make-4.4.1.tar.gz',
                'dd16fb1d67bfab79a72f5e8390735c49e3e8e70b4945a15ab1f81ddb78658fb3')
    if not (V/'make').exists():
        with tarfile.open(gnu) as t: t.extractall(V,filter='data')
        directory=V/'make-4.4.1'
        call(['./configure','--disable-dependency-tracking'],directory)
        call(['/usr/bin/make','-j4'],directory)
        shutil.copy2(directory/'make',V/'make')
    toolchain=archive('xpack-riscv-none-elf-gcc-15.2.0-1-darwin-arm64.tar.gz',
        'https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v15.2.0-1/xpack-riscv-none-elf-gcc-15.2.0-1-darwin-arm64.tar.gz',
        '6588e8351455fad8aca37551f0e5a5543f3346bfa9a837cf03cbd3bdd4989f8f')
    if not (V/'xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-g++').exists():
        with tarfile.open(toolchain) as t: t.extractall(V,filter='data')
    if 'SSL_CERT_FILE' not in os.environ and Path('/opt/homebrew/etc/openssl@3/cert.pem').exists():
        os.environ['SSL_CERT_FILE']='/opt/homebrew/etc/openssl@3/cert.pem'
    call(['python3','tensorflow/lite/micro/tools/make/download_dependencies.py',
          '--downloads_dir','tensorflow/lite/micro/tools/make/downloads','--tensorflow_root','.'],repo)
    py=ROOT.parent/'deit-tiny-inspect/.venv/bin/python'
    call([str(py),'-m','pip','install','--cache-dir',str(ROOT.parent/'deit-tiny-inspect/.pip-cache'),
          'flatbuffers==25.12.19','tflite==2.18.0'])
    print('Pinned local dependencies ready.')


if __name__=='__main__': main()
