#!/usr/bin/env python3
"""Synthetic-fixture operator verification, NOT full-model calibration."""
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT.parent / "deit-tiny-inspect/results/float-fixtures.pt"


def round_away(a):
    return np.copysign(np.floor(np.abs(a)+0.5), a)


def quant_multiplier(scale):
    q, shift = math.frexp(float(scale))
    mult = math.floor(q*(1 << 31)+0.5)
    if mult == 1 << 31:
        mult //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    assert -31 <= shift <= 30
    return mult, shift


def scale_integer(acc, multiplier, shift):
    # LiteRT/TFLite single-rounding profile, TFLITE_SINGLE_ROUNDING=1.
    # Python arbitrary-precision // is floor division, including negative values.
    total_shift = 31-shift
    value = (int(acc)*int(multiplier)+(1 << (total_shift-1))) // (1 << total_shift)
    assert -(1 << 31) <= value < (1 << 31)
    return value


def c_array(name, ctype, a):
    a = np.asarray(a).reshape(-1)
    lines = [f"static const {ctype} {name}[{a.size}] = {{"]
    for i in range(0, a.size, 16):
        lines.append("    " + ", ".join(str(int(x)) for x in a[i:i+16]) + ",")
    lines.append("};")
    return "\n".join(lines)


def main():
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    snapshot = torch.load(FIXTURE, map_location="cpu", weights_only=True)
    fixture = snapshot["operators"]["blocks.0.mlp.fc1"]
    x = fixture["input"].numpy().astype(np.float64).reshape(197,192)
    w = fixture["weight"].numpy().astype(np.float64)
    bias = fixture["bias"].numpy().astype(np.float64)
    fp_y = fixture["output"].numpy().astype(np.float64).reshape(197,768)
    sx = float(np.max(np.abs(x)) / 127)
    sw = np.max(np.abs(w), axis=1)/127
    sy = float(np.max(np.abs(fp_y))/127)
    assert sx > 0 and sy > 0 and bool(np.all(sw > 0))
    xq = np.clip(round_away(x/sx), -127, 127).astype(np.int8)
    wq = np.clip(round_away(w/sw[:,None]), -127, 127).astype(np.int8)
    bq = round_away(bias/(sx*sw)).astype(np.int64)
    assert np.max(np.abs(bq)) < (1 << 31)
    # Conservative bound holds for EVERY partial sum with |input|<=127.
    bound = 127*np.abs(wq.astype(np.int64)).sum(axis=1) + np.abs(bq)
    assert np.max(bound) <= (1 << 31)-1
    acc = xq.astype(np.int64) @ wq.astype(np.int64).T + bq
    qp = [quant_multiplier(sx*s/sy) for s in sw]
    mult = np.array([q[0] for q in qp], dtype=np.int64)
    shifts = np.array([q[1] for q in qp], dtype=np.int64)
    # Full operator reference, computed in wide host integers. Target is a tile.
    yq = np.empty(acc.shape, dtype=np.int8)
    for row in range(197):
        for channel in range(768):
            yq[row,channel] = max(-128,min(127,scale_integer(
                int(acc[row,channel]),int(mult[channel]),int(shifts[channel]))))

    m, k, n = 2, 192, 8
    # Independent arbitrary-precision scalar check for the exported tile.
    for row in range(m):
        for channel in range(n):
            scalar = int(bq[channel]) + sum(int(xq[row,i])*int(wq[channel,i]) for i in range(k))
            assert scalar == int(acc[row,channel])
    assert np.any(xq[:m] < 0) and np.any(wq[:n] < 0)
    fields = ["#include <stdint.h>", f"#define TEST_M {m}\n#define TEST_K {k}\n#define TEST_N {n}"]
    for name, ctype, data in [
        ("input_q","int8_t",xq[:m]), ("weight_q","int8_t",wq[:n]),
        ("bias_q","int32_t",bq[:n]), ("multipliers","int32_t",mult[:n]),
        ("shifts","int32_t",shifts[:n]), ("expected_acc","int32_t",acc[:m,:n]),
        ("expected_y","int8_t",yq[:m,:n])]:
        fields.append(c_array(name,ctype,data))

    # Directed ties, INT32 endpoints, saturation and deterministic random cases.
    tests = []
    for value in [-(1<<31),-(1<<31)+1,-1025,-257,-255,-5,-3,-1,0,1,3,5,255,257,1025,(1<<31)-1]:
        for multiplier, shift in [(1<<30,0),(1<<30,-1),((1<<31)-1,-31)]:
            tests.append((value,multiplier,shift))
    tests += [(17,0,0),(-17,0,0),((1<<30)-1,1<<30,1),(-(1<<30),1<<30,1)]
    rng = random.Random(6455)
    for _ in range(256):
        shift = rng.randrange(-31,3)
        limit = 1 << (31-max(0,shift))
        tests.append((rng.randrange(-limit,limit),rng.randrange(1<<30,1<<31),shift))
    fields.append("static const struct { int32_t x,m,shift,scaled; int8_t clamped; } requant_cases[] = {")
    for value, multiplier, shift in tests:
        expected = scale_integer(value,multiplier,shift)
        fields.append(f"    {{{value},{multiplier},{shift},{expected},{max(-128,min(127,expected))}}},")
    fields.append("};")
    (build/"vectors.h").write_text("\n".join(fields)+"\n")
    np.savez_compressed(build/"operator-reference.npz",input_q=xq,weight_q=wq,
        bias_q=bq,acc_reference=acc,output_q=yq,input_scale=sx,weight_scales=sw,
        output_scale=sy,multipliers=mult,shifts=shifts)
    report = dict(operator="blocks.0.mlp.fc1",model_shape=dict(M=197,K=192,N=768),
        target_tile=dict(M=m,K=k,N=n,macs=m*k*n),
        input_scale=sx,output_scale=sy,weight_scale_min=float(sw.min()),
        weight_scale_max=float(sw.max()),all_zero_points=0,
        input_range=[-127,127],weight_range=[-127,127],output_range=[-128,127],
        offline_quantization_rounding="nearest, ties away from zero",
        requant_profile="TFLite v2.19.0 TFLITE_SINGLE_ROUNDING=1, scalar int32 form",
        conservative_max_abs_acc_bound=int(bound.max()),
        requant_tests=len(tests),fixture_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        limitations=["Scales estimated from one synthetic float snapshot, not representative calibration",
                     "Full operator integer reference on host; only 2x8 tile runs on Spike",
                     "No full-model W8A8 accuracy claim or custom ISA"],
        first_output=dict(acc=int(acc[0,0]),q=int(yq[0,0]),
                          multiplier=int(mult[0]),shift=int(shifts[0])))
    (build/"reference.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"Prepared full operator reference {acc.shape}, Spike tile M={m} K={k} N={n}")
    print(f"Bound: |every partial accumulator| <= {int(bound.max()):,}; no INT32 overflow")
    print(f"Requantization tests: {len(tests)}; first output acc={acc[0,0]}, INT8={yq[0,0]}")


if __name__ == "__main__":
    main()
