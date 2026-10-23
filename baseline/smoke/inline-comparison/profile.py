#!/usr/bin/env python3
"""Count retired instructions from main entry through its return to startup."""
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parent
results = {}
for variant in ("inline", "noinline"):
    folder = root / variant
    subprocess.run(["spike", "--isa=RV64IMC", "-m32", "-l", "--log-commits",
                    "--log=spike.log", "dot.elf"], cwd=folder, check=True, timeout=30)
    symbols = subprocess.check_output(
        ["riscv64-elf-nm", "-S", "--defined-only", "dot.elf"], cwd=folder, text=True)
    ranges = {}
    for name in ("main", "dot8_t"):
        m = re.search(rf"^([0-9a-f]+) ([0-9a-f]+) T {name}$", symbols, re.M)
        assert m, name
        start, size = (int(x, 16) for x in m.groups())
        ranges[name] = (start, start + size)
    tohost = int(re.search(r"^([0-9a-f]+) D tohost$", symbols, re.M)[1], 16)
    counts = Counter()
    mix = Counter()
    begun = ended = False
    for line in (folder / "spike.log").read_text().splitlines():
        if "exception" in line or "trap_" in line:
            raise RuntimeError(line)
        # Commit records, not the separate instruction-disassembly records.
        m = re.match(r"^core\s+0:\s+3\s+0x([0-9a-f]+)\s+\(0x([0-9a-f]+)\)", line)
        if not m:
            continue
        pc, encoding = (int(x, 16) for x in m.groups())
        if pc == 0x80000000:
            begun = True
        if not begun:
            continue
        owner = next((name for name, (lo, hi) in ranges.items() if lo <= pc < hi), "startup")
        counts[owner] += 1
        if owner in ("main", "dot8_t"):
            # This kernel's loads/multiplies/branches are all 32-bit encodings.
            # Inspect the complete generated disassembly for other instruction forms.
            opcode = encoding & 0x7f
            if opcode == 0x03:
                mix["load"] += 1
            elif opcode == 0x3b and (encoding >> 25) == 1 and ((encoding >> 12) & 7) == 0:
                mix["multiply"] += 1
            elif opcode == 0x63:
                mix["conditional_branch"] += 1
            else:
                mix["other"] += 1
        store = re.search(r"mem 0x([0-9a-f]+) 0x([0-9a-f]+)", line)
        if store and int(store[1], 16) == tohost:
            assert int(store[2], 16) == 1, "Target reported failure"
            ended = True
            break
    assert begun and ended, "Missing startup or success store"
    work = counts["main"] + counts["dot8_t"]
    assert mix["load"] == 8 and mix["multiply"] == 4
    # Independent manual count from disassembly: inline 7+4*7+1;
    # noinline main 7, dot8_t 4+4*7+1. No hardware cycle inference.
    assert work == (36 if variant == "inline" else 40)
    results[variant] = dict(instructions_by_region=dict(counts),
                            main_and_callees=work, mix=dict(mix), macs=4,
                            mac_per_instruction=4/work,
                            startup_through_success_store=sum(counts.values()))
    print(variant, json.dumps(results[variant]))
(root / "profile-results.json").write_text(json.dumps(results, indent=2) + "\n")
