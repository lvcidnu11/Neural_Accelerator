#!/usr/bin/env python3
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
MASK64 = (1 << 64)-1


def packed(values):
    return sum((v & 255) << (8*i) for i,v in enumerate(values))


def lane(bits,i):
    v = (bits >> (8*i)) & 255
    return v if v < 128 else v-256


def reference(old,a,w):
    total = old + sum(lane(a,i)*lane(w,i) for i in range(8))
    low = total & 0xffffffff
    return (low if low < (1 << 31) else low-(1 << 32)) & MASK64


def main():
    BUILD.mkdir(exist_ok=True)
    rng = random.Random(6455)
    normals = [(0,packed(range(1,16,2)),packed(range(2,17,2)))]
    for a in [0,1,-1,127,-128]:
        for w in [0,1,-1,127,-128]:
            for acc in [0,0x7fffffff,0x80000000,0x12345678ffffffff]:
                normals.append((acc,packed([a]*8),packed([w]*8)))
    for i in range(8):
        a=[0]*8
        a[i]=-128
        normals.append((0,packed(a),packed(range(1,9))))
    normals += [(rng.getrandbits(64),rng.getrandbits(64),rng.getrandbits(64)) for _ in range(1024)]
    cases=[]
    for old,a,w in normals:
        cases.append((0,old,a,w,reference(old,a,w)))
    # Explicit destination/source overlap and zero-register patterns.
    for mode in range(1,9):
        for old,a,w in normals[:32] + normals[-32:]:
            actual_a = old if mode in [1,3] else a
            actual_w = old if mode in [2,3] else w
            if mode==4: actual_w=a
            if mode in [6,8]: actual_a=0
            if mode in [7,8]: actual_w=0
            expected=0 if mode==5 else reference(old,actual_a,actual_w)
            cases.append((mode,old,a,w,expected))
    text=["#include <stdint.h>","static const struct { unsigned mode; uint64_t old,a,w,expected; } instruction_cases[] = {"]
    for mode,old,a,w,expected in cases:
        values=",".join(f"UINT64_C(0x{x:016x})" for x in [old,a,w,expected])
        text.append(f"    {{{mode},{values}}},")
    text.append("};")
    (BUILD/"instruction-vectors.h").write_text("\n".join(text)+"\n")
    (BUILD/"test-manifest.json").write_text(json.dumps(dict(seed=6455,
        instruction_cases=len(cases),normal_cases=len(normals),overlap_zero_cases=len(cases)-len(normals),
        extra_dependent_qmacs=33),indent=2)+"\n")

    # Kernel K-tail / alignment cases: keep these outside the measured model tile.
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location("linear_prepare",ROOT.parent/"deit-linear-int8/prepare.py")
    ref=module_from_spec(spec)
    spec.loader.exec_module(ref)
    import numpy as np
    fields=["#include <stdint.h>"]
    rows=[]
    for k in [0,1,7,8,9,15,16,17,31,32]:
        for offset in [0,1]:
            index=len(rows)
            m,n=2,3
            x=[rng.randrange(-128,128) for _ in range(m*k)]
            w=[rng.randrange(-128,128) for _ in range(n*k)]
            bias=[rng.randrange(-1000,1001) for _ in range(n)]
            acc=[bias[c]+sum(x[r*k+i]*w[c*k+i] for i in range(k)) for r in range(m) for c in range(n)]
            out=[max(-128,min(127,ref.scale_integer(a,1<<30,-3))) for a in acc]
            for name,ctype,values in [
                (f"tx{index}","int8_t",[0]*offset+(x or [0])),
                (f"tw{index}","int8_t",[0]*offset+(w or [0])),
                (f"tb{index}","int32_t",bias),
                (f"ta{index}","int32_t",acc),
                (f"ty{index}","int8_t",out)]:
                declaration=ref.c_array(name,ctype,np.array(values))
                if name.startswith(("tx","tw")):
                    declaration="_Alignas(8) "+declaration
                fields.append(declaration)
            rows.append(f"{{{m},{k},{n},tx{index}+{offset},tw{index}+{offset},tb{index},ta{index},ty{index}}}")
    fields.append("static const struct { int m,k,n; const int8_t *x,*w; const int32_t *bias,*acc; const int8_t *y; } tail_cases[] = {")
    fields.append(",\n".join(rows)+"\n};")
    (BUILD/"tail-vectors.h").write_text("\n".join(fields)+"\n")
    print(f"Prepared {len(cases)} ISA cases + dependent chain; {len(rows)} tail/alignment kernel cases")


if __name__ == "__main__":
    main()
