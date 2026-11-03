#!/usr/bin/env python3
"""Full-model X1: matched B1 software, qmac8, software requantization."""
import importlib.util,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;WS=HERE.parents[1];REPO=WS
spec=importlib.util.spec_from_file_location('evaluate',WS/'experiments/tpu-evaluation-2026-11-02/evaluate.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
original=base.kernel
base.VARIANTS['x1-qmac']=(False,True,False)
def kernel(src,inline,hoist):
 src=original(src,inline,hoist)
 a=src.index('inline int8_t requant(');b=src.index('\ninline int32_t low32',a)
 block=src[a:b]
 old='#ifdef __riscv\n  return requant8_int8(acc,requant8_params(multiplier,shift,0));\n#else\n'
 assert block.count(old)==1 and block.count('#endif')==1
 block=block.replace(old,'').replace('#endif\n','')
 return src[:a]+block+src[b:]
def main():
    base.kernel=kernel
    out=HERE/'build';out.mkdir(exist_ok=True)
    r=base.evaluate(REPO,out,'x1-qmac',False)
    folder=out/'x1-qmac';asm=base.run(['riscv64-elf-objdump','-d','-M','no-aliases',folder/'riscv/program.elf'],folder/'disassembly.log')
    words=[int(m[1],16) for line in asm.splitlines() if (m:=re.match(r'^\s*[0-9a-f]+:\s+([0-9a-f]{4,8})\s+',line))]
    counts={'qmac8':sum((w&0xfe00707f)==0x4000000b for w in words),'requant8':sum((w&0xfe00707f)==0x4200000b for w in words)}
    assert counts['qmac8']>0 and counts['requant8']==0,counts
    r.update(definition='B1 software optimizations + qmac8 and software requantization',static_opcode_counts=counts)
    (folder/'result.json').write_text(json.dumps(r,indent=2)+'\n')
    print('PASS X1',r['retired'],counts,flush=True)

if __name__=="__main__":
    main()
