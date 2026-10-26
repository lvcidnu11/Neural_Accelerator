#!/usr/bin/env python3
from collections import Counter
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
BUILD=ROOT/"build"
MATCH,MASK=0x4000000b,0xfe00707f
MATCH_REQUANT=0x4200000b


def commits(path):
    for line in path.read_text().splitlines():
        if "exception" in line or "trap_" in line:
            raise RuntimeError(line)
        m=re.match(r"^core\s+0:\s+3\s+0x([0-9a-f]+)\s+\(0x([0-9a-f]+)\)",line)
        if m:
            yield int(m[1],16),int(m[2],16),line


def category(m):
    name=m.removeprefix("c.")
    if name=="qmac8": return "custom_mac"
    if name=="requant8": return "custom_requant"
    if name in ["lb","lbu","lh","lhu","lw","lwu","ld","lwsp","ldsp"]: return "load"
    if name in ["sb","sh","sw","sd","swsp","sdsp"]: return "store"
    if name.startswith("mul"): return "multiply"
    if name in ["beq","bne","blt","bge","bltu","bgeu","beqz","bnez"]: return "conditional_branch"
    if name in ["jal","jalr","j","jr"]: return "jump_return"
    return "integer_other"


def measure(variant):
    folder=BUILD/variant
    symbols={}
    for line in (folder/"symbols.txt").read_text().splitlines():
        fields=line.split()
        symbols[fields[-1]]=int(fields[0],16)
    static={}
    for line in (folder/"program.disasm").read_text().splitlines():
        m=re.match(r"\s*([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)",line)
        if m: static[int(m[1],16)]=m[2]
    active=False
    begins=ends=0
    success=False
    mix=Counter()
    mnemonics=Counter()
    for pc,bits,line in commits(folder/"spike.log"):
        if pc==symbols["kernel_begin"]:
            assert not active
            active=True
            begins+=1
        if pc==symbols["kernel_end"]:
            assert active
            active=False
            ends+=1
        if active:
            if bits & MASK==MATCH:
                name="qmac8"
            elif bits & MASK==MATCH_REQUANT:
                name="requant8"
            else:
                name=static[pc]
            mnemonics[name]+=1
            mix[category(name)]+=1
        store=re.search(r"mem 0x([0-9a-f]+) 0x([0-9a-f]+)",line)
        if store and int(store[1],16)==symbols["tohost"]:
            assert int(store[2],16)==1
            success=True
            break
    assert success and begins==ends==1 and not active
    expected_custom=384 if variant!="scalar" else 0
    expected_requant=16 if variant=="qmac8_requant" else 0
    assert mnemonics["qmac8"]==expected_custom
    assert mnemonics["requant8"]==expected_requant
    if variant=="scalar": assert mnemonics["mulw"]==3072
    section_sizes={}
    for line in (folder/"sections.txt").read_text().splitlines():
        fields=line.split()
        if len(fields)==3 and fields[0].startswith("."):
            section_sizes[fields[0]]=int(fields[1])
    count=sum(mix.values())
    return dict(retired_instructions=count,mathematical_macs=3072,
        mac_per_retired_instruction=3072/count,qmac8_instructions=expected_custom,
        requant8_instructions=expected_requant,
        custom_instructions=expected_custom+expected_requant,
        custom_frequency=(expected_custom+expected_requant)/count,
        memory_instructions_per_mac=(mix["load"]+mix["store"])/3072,
        mix=dict(mix),mnemonics=dict(mnemonics),section_bytes=section_sizes)


def main():
    tests=json.loads((BUILD/"test-manifest.json").read_text())
    actual=sum(1 for _,bits,_ in commits(BUILD/"instruction-tests/spike.log") if bits & MASK==MATCH)
    assert actual==tests["instruction_cases"]+tests["extra_dependent_qmacs"],actual
    reqtests=json.loads((BUILD/"requant-test-manifest.json").read_text())
    actual_req=sum(1 for _,bits,_ in commits(BUILD/"requant-tests/spike.log") if bits & MASK==MATCH_REQUANT)
    assert actual_req==reqtests["instruction_cases"]+reqtests["scalar_reference_cases"],actual_req
    scalar=measure("scalar")
    custom=measure("qmac8")
    combined=measure("qmac8_requant")
    ratio=scalar["retired_instructions"]/custom["retired_instructions"]
    total_ratio=scalar["retired_instructions"]/combined["retired_instructions"]
    incremental_ratio=custom["retired_instructions"]/combined["retired_instructions"]
    previous=json.loads((ROOT.parent/"qmac8-spike/build/comparison.json").read_text())
    assert scalar["retired_instructions"]==previous["scalar"]["retired_instructions"]
    assert custom["retired_instructions"]==previous["qmac8"]["retired_instructions"]
    report=dict(scalar=scalar,qmac8=custom,qmac8_requant=combined,qmac_instruction_count_ratio=ratio,
        instruction_reduction_fraction=1-custom["retired_instructions"]/scalar["retired_instructions"],
        combined_instruction_count_ratio=total_ratio,
        requant_incremental_count_ratio=incremental_ratio,
        requant_incremental_reduction=1-combined["retired_instructions"]/custom["retired_instructions"],
        combined_reduction=1-combined["retired_instructions"]/scalar["retired_instructions"],
        isa_test_qmacs=actual,isa_test_requant8=actual_req,
        original_baseline_counts_reproduced=True,
        measurement="Same model-derived 2x192 by 8x192 tile, driver, input, zero points, bias, exact single-rounding/clamp rules, accumulator/output stores, compiler flags, linker and Spike binary; runtime markers include callees",
        parameter_packing="Runtime packing of multiplier/shift/zp is INSIDE the measured interval; no prepacked metadata shortcut",
        cycles_measured=False,full_model=False,requantization_accelerated=True)
    (BUILD/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"Retired instructions: scalar {scalar['retired_instructions']:,} -> qmac8 {custom['retired_instructions']:,} -> qmac8+requant8 {combined['retired_instructions']:,}")
    print(f"Requant incremental reduction {report['requant_incremental_reduction']:.2%}; count ratio {incremental_ratio:.3f}x")
    print(f"Combined versus scalar: {total_ratio:.3f}x count ratio; reduction {report['combined_reduction']:.2%}")
    print(f"Measured {combined['qmac8_instructions']} qmac8 + {combined['requant8_instructions']} requant8")
    print(f"ISA tests executed {actual} qmac8 and {actual_req} requant8 instructions")
    print("ISA-level counts only: no cycle speedup, Fmax or full-model speedup measured.")


if __name__ == "__main__":
    main()
