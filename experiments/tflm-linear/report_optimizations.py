#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parent
B=ROOT/'build'
def main():
    records=[];paths=[]
    for shape in ['2x8','197x768']:
        for block in [1,2,4]:
            for fused in [False,True]:
                folder=B/(shape+'-gelu'+('-fused' if fused else ''))/('riscv-custom'+('-kernel' if block==1 else f'-b{block}'))
                p=json.loads((folder/'profile.json').read_text());records.append(p)
                assert p['elf_sha256']==hashlib.sha256((folder/'program.elf').read_bytes()).hexdigest()
                assert p['correctness']['outputs_checked']==(16 if shape=='2x8' else 151296)
                assert p['correctness']['failure_stage']==0
                paths += [folder/n for n in ['program.elf','build.json','profile.json']]
    (B/'optimizations.json').write_text(json.dumps(records,indent=2)+'\n')
    full=[p for p in records if p['scope'].startswith('Full')]
    baseline=next(p['invoke_retired'] for p in full if p['blocking_factor']==1 and not p['fused'])
    lines=['# Fusion and register blocking — 2026-10-25','',
        'Same mathematical fc1 -> INT8 requantization -> GELU mapping and every final output checked.',
        'Block 1 also uses the SAME project::linear kernel as blocks 2/4, so fusion is isolated from the earlier kernel refactor.','',
        '| Blocking factor | Separate fc1/GELU | Fused fc1/GELU |','|---|---:|---:|']
    for block in [1,2,4]:
        plain=next(p for p in full if p['blocking_factor']==block and not p['fused'])
        fused=next(p for p in full if p['blocking_factor']==block and p['fused'])
        lines.append(f"| {block} | {plain['invoke_retired']:,} | {fused['invoke_retired']:,} |")
    best=next(p for p in full if p['blocking_factor']==4 and p['fused'])
    lines+=['',f"Combined reduction versus matched block-1 kernel: {100*(1-best['invoke_retired']/baseline):.4f}%.",
        'Earlier TFLM adapter count was 27,551,040; comparison against that count also includes kernel-refactor effects.',
        'Arena used: separate 309,584 bytes, fused 196,000 bytes. All final 151,296 outputs match.',
        'Unfused runs also verify fc1 intermediates. Fused runs avoid that intermediate allocation/store and verify final outputs.',
        '640 directed dimension/tail/alignment/fusion cases pass host UBSan and actual Spike.',
        'No hardware cycles or physical performance is claimed.','']
    asm=(B/'197x768-gelu/riscv-custom-b4/program.disasm').read_text()
    inst={}
    for line in asm.splitlines():
        m=re.match(r'^\s*([0-9a-f]+):\s+([0-9a-f]{4,8})\s+(\S+)\s*(.*)',line)
        if m: inst[int(m[1],16)]=(int(m[2],16),m[3],m[4],line)
    loop=None
    for pc,(word,name,ops,line) in inst.items():
        match=re.search(r',([0-9a-f]+) <',ops)
        if name!='bne' or not match:continue
        start=int(match[1],16)
        if start>=pc:continue
        body=[data for addr,data in inst.items() if start<=addr<=pc]
        qmac=sum((data[0]&0xfe00707f)==0x4000000b for data in body)
        if qmac==4:
            assert sum(data[1] in ['ld','c.ld'] for data in body)==5
            assert all('(sp)' not in data[2] for data in body)
            loop=body;break
    assert loop is not None
    (B/'blocking4-hot-loop.txt').write_text('\n'.join(data[3] for data in loop)+'\n')
    lines+=['Verified disassembly: the block-4 hot loop has 5 packed loads and 4 qmac8 instructions,',
            'with no stack access inside the loop. Function entry/exit and outer loops can still access the stack.','']
    (B/'optimizations.md').write_text('\n'.join(lines));print('\n'.join(lines))
    files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.cc','.h','.S','.ld','.md']]+paths
    for shape in ['2x8-gelu','197x768-gelu','2x8-gelu-fused','197x768-gelu-fused']:
        files += [B/shape/n for n in ['model.tflite','reference.json','model_data.h']]
    (B/'optimizations-manifest.json').write_text(json.dumps({str(p.relative_to(ROOT.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
if __name__=='__main__':main()
