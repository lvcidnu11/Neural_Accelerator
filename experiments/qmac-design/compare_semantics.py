#!/usr/bin/env python3
"""Provisional instruction arithmetic models, NOT a Spike implementation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1


def signed(bits, width):
    bits &= (1 << width) - 1
    return bits - (1 << width) if bits & (1 << (width-1)) else bits


def pack(values):
    return sum((v & 255) << (8*i) for i, v in enumerate(values))


def dot_from_registers(a, w):
    # Register lane extraction + signed interpretation, separate from the
    # vector generator's arithmetic over Python lists.
    total = 0
    for shift in range(0, 64, 8):
        total += signed(a >> shift, 8) * signed(w >> shift, 8)
    assert -(1 << 18) <= total < (1 << 18)  # Fits signed 19 bits.
    return total


def qmac32(old_rd, rs1, rs2):
    value = signed(old_rd, 32) + dot_from_registers(rs1, rs2)
    return signed(value, 32) & MASK64


def qdot(rs1, rs2):
    return dot_from_registers(rs1, rs2) & MASK64


def addw(rs1, rs2):
    return signed((rs1 & MASK32) + (rs2 & MASK32), 32) & MASK64


def main():
    data = json.loads((ROOT / "vectors.json").read_text())
    rows = []
    wrapped = 0
    for case in data["cases"]:
        a, w = pack(case["a"]), pack(case["w"])
        # Deliberately arbitrary high bits: provisional accumulator semantics
        # read only old_rd[31:0], then sign-extend the returned 32-bit result.
        old_rd = (0x12345678 << 32) | (case["acc"] & MASK32)
        one = qmac32(old_rd, a, w)
        two = addw(old_rd, qdot(a, w))
        expected = signed(case["expected_bits"], 32) & MASK64
        assert one == two == expected, case["label"]
        if not -(1 << 31) <= case["mathematical_sum"] < (1 << 31):
            wrapped += 1
        if not case["label"].startswith("random_"):
            rows.append(dict(label=case["label"], signed_result=signed(one, 64),
                             qmac_rd_hex=f"0x{one:016x}",
                             qdot_then_addw_rd_hex=f"0x{two:016x}"))
    out = dict(scope="Arithmetic models only; no instruction encoding or Spike execution",
               seed=data["seed"], passed_cases=len(data["cases"]),
               cases_with_int32_wrap=wrapped, directed_results=rows)
    (ROOT / "results/semantics-comparison.json").write_text(json.dumps(out, indent=2)+"\n")
    print(f"PASS: qmac32 == qdot + addw == independent reference, "
          f"{out['passed_cases']} cases, including {wrapped} INT32-wrap cases.")


if __name__ == "__main__":
    main()
