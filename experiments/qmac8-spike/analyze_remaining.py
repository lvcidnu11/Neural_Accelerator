#!/usr/bin/env python3
"""Attribute the existing measured trace; no new instruction is implemented."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/"build/qmac8"
symbols={}
ranges={}
for line in (FOLDER/"symbols.txt").read_text().splitlines():
    fields=line.split()
    symbols[fields[-1]]=int(fields[0],16)
    if len(fields)==4 and fields[2] in ("t","T"):
        start,size=int(fields[0],16),int(fields[1],16)
        ranges[fields[-1]]=(start,start+size)
static=[]
for line in (FOLDER/"program.disasm").read_text().splitlines():
    m=re.match(r"\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)",line)
    if m:
        static.append((int(m[1],16),int(m[2],16),m[3]))
indices=[i for i,(_,bits,_) in enumerate(static) if bits & 0xfe00707f==0x4000000b]
assert len(indices)==1
index=indices[0]
loop=static[index-2:index+4]
assert [x[2] for x in loop]==["c.ld","c.ld",".insn","c.addi","c.addi","bne"], "Loop structure changed: recheck attribution"
loop_pcs={x[0] for x in loop}
counts=Counter()
by_function=Counter()
active=False
for line in (FOLDER/"spike.log").read_text().splitlines():
    m=re.match(r"^core\s+0:\s+3\s+0x([0-9a-f]+)\s+\(0x",line)
    if not m:
        continue
    pc=int(m[1],16)
    if pc==symbols["kernel_begin"]:
        active=True
    if pc==symbols["kernel_end"]:
        active=False
        break
    if not active:
        continue
    owner=next((name for name,(lo,hi) in ranges.items() if lo<=pc<hi),"unknown")
    assert owner!="unknown"
    by_function[owner]+=1
    if pc in loop_pcs:
        counts["packed_loop"]+=1
    elif owner=="multiply_quantized":
        counts["requant_scaling_helper"]+=1
    elif owner=="clamp_int8":
        counts["requant_clamp_helper"]+=1
    else:
        counts["setup_outer_control_memory_calls"]+=1
total=sum(counts.values())
comparison=json.loads((ROOT/"build/comparison.json").read_text())
assert total==comparison["qmac8"]["retired_instructions"]
assert counts["packed_loop"]==384*6
report=dict(total=total,counts=dict(counts),
    fractions={key:value/total for key,value in counts.items()},
    by_function=dict(by_function),
    note="Scaling/clamp helper counts include their returns but exclude call-site argument/call setup, which is in the remaining group. This is instruction attribution, not cycle attribution.",
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    trace_sha256=hashlib.sha256((FOLDER/"spike.log").read_bytes()).hexdigest())
(ROOT/"build/remaining-analysis.json").write_text(json.dumps(report,indent=2)+"\n")
for name,count in counts.items():
    print(f"{name}: {count:,} ({count/total:.2%})")
print(f"Total: {total:,}; no new ISA or cycle measurement")
