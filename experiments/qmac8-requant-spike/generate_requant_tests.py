#!/usr/bin/env python3
"""Arbitrary-precision independent requant8 reference and register tests."""
import json
from pathlib import Path
import random
import re

ROOT=Path(__file__).resolve().parent
BUILD=ROOT/"build"
MASK64=(1<<64)-1


def signed(bits,width):
    bits &= (1<<width)-1
    return bits-(1<<width) if bits & (1<<(width-1)) else bits


def pack(mult,shift,zp):
    return mult | ((shift & 255)<<32) | ((zp & 255)<<40)


def reference(x,params):
    mult=params & 0xffffffff
    shift=signed(params>>32,8)
    zp=signed(params>>40,8)
    assert mult<=0x7fffffff and -31<=shift<=30 and params>>48==0
    bits=31-shift
    scaled=(signed(x,32)*mult+(1<<(bits-1)))//(1<<bits)
    output=max(-128,min(127,scaled+zp))
    return output & MASK64


def main():
    rng=random.Random(246455)
    normals=[]
    # Exactly the existing 308 scalar reference cases, not a new rounding rule.
    header=(ROOT.parent/"deit-linear-int8/build/vectors.h").read_text()
    section=header.split("requant_cases[] = {",1)[1]
    baseline=[]
    for line in section.splitlines():
        m=re.fullmatch(r"\s*\{(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+)\},",line)
        if m:
            x,mult,shift,scaled,output=map(int,m.groups())
            p=pack(mult,shift,0)
            assert reference(x,p)==output & MASK64
            normals.append((x & MASK64,p))
            baseline.append((x,mult,shift,output))
    assert len(baseline)==308
    for x in [-(1<<31),-(1<<31)+1,-1025,-257,-255,-5,-3,-1,0,1,3,5,255,257,1025,(1<<31)-1]:
        for mult in [0,1<<30,(1<<31)-1]:
            for shift in [-31,-9,-1,0,1,30]:
                for zp in [-128,0,127]:
                    # Deliberately poison rs1 high bits; they must be ignored.
                    normals.append(((rng.getrandbits(32)<<32)|(x & 0xffffffff),pack(mult,shift,zp)))
    normals += [(rng.getrandbits(64),pack(rng.randrange(1<<31),rng.randrange(-31,31),rng.randrange(-128,128)))
                for _ in range(1024)]
    rows=[(0,x,p,reference(x,p)) for x,p in normals]
    for mode in range(1,9):
        for x,p in normals[:32]+normals[-32:]:
            actual_x=p if mode in [3,4] else x
            actual_p=0 if mode==6 else p
            if mode in [5,8]: actual_x=0
            if mode==8: actual_p=0
            expected=0 if mode in [7,8] else reference(actual_x,actual_p)
            rows.append((mode,x,p,expected))
    out=["#include <stdint.h>","static const struct { int32_t x,m,shift; int8_t expected; } scalar_requant_cases[] = {"]
    out += [f"    {{{x},{m},{s},{y}}}," for x,m,s,y in baseline]
    out += ["};","static const struct { unsigned mode; uint64_t x,params,expected; } requant_instruction_cases[] = {"]
    for mode,x,p,y in rows:
        out.append(f"    {{{mode},UINT64_C(0x{x:016x}),UINT64_C(0x{p:016x}),UINT64_C(0x{y:016x})}},")
    out.append("};")
    (BUILD/"requant-vectors.h").write_text("\n".join(out)+"\n")
    (BUILD/"requant-test-manifest.json").write_text(json.dumps(dict(seed=246455,
        scalar_reference_cases=len(baseline),normal_cases=len(normals),
        alias_and_zero_cases=len(rows)-len(normals),instruction_cases=len(rows)),indent=2)+"\n")
    print(f"Prepared {len(rows)} requant8 ISA cases, including {len(baseline)} existing scalar references")


if __name__=="__main__":
    main()
