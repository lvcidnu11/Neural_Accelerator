#!/usr/bin/env python3
"""Count commits between runtime markers, including called requant helpers."""
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
symbols = {}
for line in (BUILD/"symbols.txt").read_text().splitlines():
    fields = line.split()
    if fields:
        symbols[fields[-1]] = int(fields[0],16)
begin, end = symbols["kernel_begin"], symbols["kernel_end"]
instructions = {}
for line in (BUILD/"linear.disasm").read_text().splitlines():
    match = re.match(r"\s*([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)",line)
    if match:
        instructions[int(match[1],16)] = match[2]
count = Counter()
active = False
begins = ends = 0
success = False
for line in (BUILD/"spike.log").read_text().splitlines():
    if "exception" in line or "trap_" in line:
        raise RuntimeError(line)
    match = re.match(r"^core\s+0:\s+3\s+0x([0-9a-f]+)\s+\(0x",line)
    if not match:
        continue
    pc = int(match[1],16)
    if pc == begin:
        assert not active
        active = True
        begins += 1
    if pc == end:
        assert active
        active = False
        ends += 1
    if active:
        count[instructions[pc]] += 1
    store = re.search(r"mem 0x([0-9a-f]+) 0x([0-9a-f]+)",line)
    if store and int(store[1],16) == symbols["tohost"]:
        assert int(store[2],16) == 1
        success = True
        break
assert success and begins == ends == 1 and not active
reference = json.loads((BUILD/"reference.json").read_text())
macs = reference["target_tile"]["macs"]
assert count["mulw"] == macs
assert not any(name.startswith(("fadd","fmul","fdiv","flw","fld","fsw","fsd")) for name in count)
report = dict(retired_instructions=sum(count.values()),mathematical_macs=macs,
    mac_per_retired_instruction=macs/sum(count.values()),mnemonic_mix=dict(count),
    interval="kernel_begin inclusive to kernel_end exclusive, includes argument setup, function/callee prologues, dot, bias, requantization, stores and returns; excludes correctness tests, startup and exit",
    cycles_measured=False, custom_instructions=False)
(BUILD/"profile.json").write_text(json.dumps(report,indent=2)+"\n")
print(f"Measured tile: {macs:,} MACs, {sum(count.values()):,} retired instructions, "
      f"{report['mac_per_retired_instruction']:.6f} MAC/instruction")
