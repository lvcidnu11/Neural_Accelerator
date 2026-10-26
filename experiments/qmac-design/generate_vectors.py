#!/usr/bin/env python3
"""Independent arbitrary-precision host reference and deterministic test vectors."""
import json
from pathlib import Path
import random

root = Path(__file__).resolve().parent
cases = []


def add(label, a, w, acc=0):
    assert len(a) == len(w) == 8
    assert all(-128 <= x <= 127 for x in a + w)
    wide = acc + sum(x*y for x, y in zip(a, w))
    expected = wide & 0xffffffff
    cases.append(dict(label=label, a=a, w=w, acc=acc,
                      expected_bits=expected, mathematical_sum=wide))


add("course_positive_744", list(range(1, 16, 2)), list(range(2, 17, 2)))
add("all_zero", [0]*8, [-128]*8)
add("max_positive_dot_131072", [-128]*8, [-128]*8)
add("all_positive_max", [127]*8, [127]*8)
add("max_negative_dot", [-128]*8, [127]*8)
add("negative_one", [-1]*8, [1]*8)
add("cancellation", [-128,127]*4, [1]*8)
add("positive_acc_wrap", [-128]*8, [-128]*8, 2147483647)
add("negative_acc_wrap", [-128]*8, [127]*8, -2147483648)
add("preserve_negative_acc", [0]*8, [127]*8, -2147483648)
for lane in range(8):
    a = [0]*8
    a[lane] = -128
    add(f"lane_{lane}", a, [1,2,3,4,5,6,7,8])
rng = random.Random(6455)
for i in range(1024):
    add(f"random_{i}", [rng.randrange(-128,128) for _ in range(8)],
        [rng.randrange(-128,128) for _ in range(8)],
        rng.randrange(-2147483648,2147483648))


def pack(lanes):
    return sum((x & 0xff) << (8*i) for i, x in enumerate(lanes))


(root / "vectors.mem").write_text("".join(
    f"{pack(c['a']):016x}{pack(c['w']):016x}{c['acc'] & 0xffffffff:08x}{c['expected_bits']:08x}\n"
    for c in cases))
(root / "vectors.json").write_text(json.dumps(dict(seed=6455, cases=cases), indent=2)+"\n")
(root / "vector_count.vh").write_text(f"`define VECTOR_COUNT {len(cases)}\n")
print(f"Generated {len(cases)} cases; seed=6455")
