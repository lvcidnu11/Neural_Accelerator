#!/usr/bin/env python3
"""Build/load a real Spike custom extension, verify it, compare a fixed tile."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
BUILD=ROOT/"build"
BASE=ROOT.parent/"deit-linear-int8"
commands=[]
SIM=os.environ.get("RISCV_SIM","spike")
CC=os.environ.get("RISCV_CC","riscv64-elf-gcc")
PYTHON=os.environ.get("MODEL_PYTHON",str(ROOT.parent/"deit-tiny-inspect/.venv/bin/python"))
FLAGS=["-march=rv64imc","-mabi=lp64","-mcmodel=medany","-ffreestanding","-O2","-fno-inline","-Wall","-Wextra","-Werror"]


def run(args,log=None,timeout=60):
    args=list(map(str,args))
    commands.append(dict(cwd=str(ROOT),argv=args))
    p=subprocess.run(args,cwd=ROOT,text=True,stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT,timeout=timeout)
    if log:
        (ROOT/log).write_text(p.stdout)
    if p.returncode:
        raise RuntimeError(f"Command failed {p.returncode}: {args}\n{p.stdout}")
    return p.stdout


def compile_target(source,name,kernel=None,custom=False,requant=False):
    folder=BUILD/name
    folder.mkdir(exist_ok=True)
    run([CC]+FLAGS+["-I.","-Ibuild","-c",source,"-o",folder/"main.o"],folder.relative_to(ROOT)/"compile-main.log")
    objects=[BUILD/"start.o",folder/"main.o"]
    if kernel:
        extra=["-DUSE_QMAC8"] if custom else []
        if requant: extra.append("-DUSE_REQUANT8")
        run([CC]+FLAGS+extra+["-I.","-c",kernel,"-o",folder/"kernel.o"],folder.relative_to(ROOT)/"compile-kernel.log")
        objects.append(folder/"kernel.o")
    run([CC]+FLAGS[:2]+["-nostdlib","-nostartfiles",f"-Wl,--no-relax,-Map={folder/'program.map'}",
        "-T",BUILD/"link.ld"]+objects+["-o",folder/"program.elf"],folder.relative_to(ROOT)/"link.log")
    run(["riscv64-elf-objdump","-d","-M","no-aliases",folder/"program.elf"],folder.relative_to(ROOT)/"program.disasm")
    run(["riscv64-elf-nm","-n","-S","--defined-only",folder/"program.elf"],folder.relative_to(ROOT)/"symbols.txt")
    run(["riscv64-elf-size","-A",folder/"program.elf"],folder.relative_to(ROOT)/"sections.txt")
    return folder


def main():
    BUILD.mkdir(exist_ok=True)
    if not all((BASE/"build"/name).exists() for name in ["vectors.h","start.S","link.ld","reference.json"]):
        print(run(["sh",BASE/"run.sh"]))
    original=ROOT.parent/"qmac8-spike"
    if not (original/"build/comparison.json").exists():
        print(run([sys.executable,original/"run.py"],timeout=180))
    print(run([PYTHON,"generate_tests.py"]).strip())
    print(run([sys.executable,"generate_requant_tests.py"]).strip())
    # Common data/layout and driver in both benchmark configurations.
    vectors=(BASE/"build/vectors.h").read_text()
    vectors=vectors.replace("static const int8_t input_q", "_Alignas(8) static const int8_t input_q")
    vectors=vectors.replace("static const int8_t weight_q", "_Alignas(8) static const int8_t weight_q")
    (BUILD/"vectors.h").write_text(vectors)
    for source in ["start.S","link.ld","reference.json"]:
        shutil.copyfile(BASE/"build"/source,BUILD/source)
    run([CC]+FLAGS[:2]+["-c",BUILD/"start.S","-o",BUILD/"start.o"])
    prefix=Path(os.environ.get("SPIKE_PREFIX",str(Path(shutil.which(SIM)).resolve().parent.parent)))
    suffix="dylib" if sys.platform=="darwin" else "so"
    library=BUILD/f"libqmac8.{suffix}"
    linkflags=["-dynamiclib","-undefined","dynamic_lookup"] if sys.platform=="darwin" else ["-shared"]
    run([os.environ.get("CXX","c++"),"-std=c++17","-O2","-Wall","-Wextra","-Werror","-fPIC"]+
        linkflags+[f"-I{prefix/'include'}","extension.cc","-o",library],"build/extension-build.log")
    ext=[f"--extlib={library}","--extension=xqmac8"]
    run([SIM,"--help"],"build/spike-version.txt")
    run([CC,"--version"],"build/compiler-version.txt")
    run([os.environ.get("CXX","c++"),"--version"],"build/host-compiler-version.txt")
    isa=compile_target("instruction_tests.c","instruction-tests")
    for mode in ["fast","logged"]:
        trace=["-l","--log-commits",f"--log={isa/'spike.log'}"] if mode=="logged" else []
        run([SIM,"--isa=RV64IMC","-m32"]+ext+trace+[isa/"program.elf"],f"build/instruction-tests/{mode}.txt")
    print("PASS: actual qmac8 ISA tests in fast and commit-logged Spike modes")
    reqtests=compile_target("requant_tests.c","requant-tests",kernel="kernel.c")
    for mode in ["fast","logged"]:
        trace=["-l","--log-commits",f"--log={reqtests/'spike.log'}"] if mode=="logged" else []
        run([SIM,"--isa=RV64IMC","-m32"]+ext+trace+[reqtests/"program.elf"],f"build/requant-tests/{mode}.txt")
    print("PASS: actual requant8 ISA and software-reference tests in fast/logged modes")
    for variant in ["qmac8","qmac8_requant"]:
        tails=compile_target("tail_tests.c",f"tail-tests-{variant}",kernel="kernel.c",custom=True,requant=variant=="qmac8_requant")
        run([SIM,"--isa=RV64IMC","-m32"]+ext+[tails/"program.elf"],f"build/tail-tests-{variant}/spike-output.txt")
    print("PASS: 20 kernel K-tail/alignment cases in both custom configurations")
    for variant in ["scalar","qmac8","qmac8_requant"]:
        folder=compile_target("main.c",variant,kernel="kernel.c",custom=variant!="scalar",requant=variant=="qmac8_requant")
        # Use the SAME extension-loaded simulator for the controlled comparison.
        # scalar has no custom instruction; an additional stock run checks that.
        run([SIM,"--isa=RV64IMC","-m32"]+ext+[folder/"program.elf"],f"build/{variant}/fast.txt")
        run([SIM,"--isa=RV64IMC","-m32"]+ext+["-l","--log-commits",f"--log={folder/'spike.log'}",folder/"program.elf"],f"build/{variant}/logged.txt")
        if variant=="scalar":
            run([SIM,"--isa=RV64IMC","-m32",folder/"program.elf"],"build/scalar/stock-spike.txt")
        print(f"PASS: {variant} Linear tile, accumulator/INT8 output and 308 requant tests")
    # Bounded negative controls: no extension and a reserved funct3 with extension.
    (BUILD/"reserved.c").write_text('int main(void) { __asm__ volatile(".word 0x4000100b"); return 0; }\n')
    reserved=compile_target(BUILD/"reserved.c","reserved")
    for name,elf,extras in [("missing-extension",isa/"program.elf",[]),
                            ("reserved-encoding",reserved/"program.elf",ext)]:
        logfile=BUILD/f"{name}.log"
        run([SIM,"--isa=RV64IMC","-m32","--instructions=2000"]+extras+
            ["-l",f"--log={logfile}",elf],f"build/{name}-output.txt",timeout=15)
        assert "trap_illegal_instruction" in logfile.read_text(),name
    print("PASS: missing extension and reserved encoding trap as illegal instructions")
    invalids={"reserved-requant-encoding":None,
              "negative-multiplier":0x80000000,
              "reserved-parameter-bits":1<<48,
              "shift-below-range":0xe000000000,
              "shift-above-range":0x1f00000000}
    for name,params in invalids.items():
        source=BUILD/f"{name}.c"
        if params is None:
            source.write_text('int main(void) { __asm__ volatile(".word 0x4200100b"); return 0; }\n')
        else:
            source.write_text('#include "requant8.h"\nint main(void) { (void)requant8_bits(123,UINT64_C(0x%016x)); return 0; }\n'%params)
        target=compile_target(source,name)
        logfile=BUILD/f"{name}.log"
        run([SIM,"--isa=RV64IMC","-m32","--instructions=2000"]+ext+
            ["-l",f"--log={logfile}",target/"program.elf"],f"build/{name}-output.txt",timeout=15)
        assert "trap_illegal_instruction" in logfile.read_text(),name
    missing_source=BUILD/"requant-missing-extension.c"
    missing_source.write_text('#include "requant8.h"\nint main(void) { (void)requant8_bits(123,0); return 0; }\n')
    missing=compile_target(missing_source,"requant-missing-extension")
    logfile=BUILD/"requant-missing-extension.log"
    run([SIM,"--isa=RV64IMC","-m32","--instructions=2000","-l",f"--log={logfile}",missing/"program.elf"],"build/requant-missing-extension-output.txt",timeout=15)
    assert "trap_illegal_instruction" in logfile.read_text()
    print("PASS: requant8 reserved encoding/parameters and missing-extension negative controls")
    print(run([sys.executable,"profile.py"]).strip())
    tracked=[ROOT/name for name in ["extension.cc","qmac8.h","kernel.c","kernel.h","main.c",
        "instruction_tests.c","tail_tests.c","generate_tests.py","run.py","profile.py",
        "requant8.h","requant_tests.c","generate_requant_tests.py"]]
    tracked += [BUILD/name for name in ["vectors.h","instruction-vectors.h","tail-vectors.h","requant-vectors.h","start.S","link.ld"]]
    tracked += [BASE/"prepare.py",BASE/"build/vectors.h",BASE/"build/reference.json",
                original/"build/comparison.json"]
    headers=[prefix/"include/riscv"/name for name in ["extension.h","processor.h","decode.h","disasm.h"]]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked+headers}
    hashes[str(Path(shutil.which(SIM)).resolve())]=hashlib.sha256(Path(shutil.which(SIM)).read_bytes()).hexdigest()
    hashes[str(library)]=hashlib.sha256(library.read_bytes()).hexdigest()
    (BUILD/"manifest.json").write_text(json.dumps(dict(commands=commands,sha256=hashes,
        simulator_prefix=str(prefix),extension_library=str(library)),indent=2)+"\n")


if __name__ == "__main__":
    main()
