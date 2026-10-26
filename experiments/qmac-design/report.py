#!/usr/bin/env python3
"""Summarize measured tests and generic synthesis, without claiming physical area."""
from collections import Counter
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
simulation = (root / "results/simulation.log").read_text()
m = re.search(r"PASS: (\d+) vectors x 3 designs; latency LANES=1:(\d+), LANES=2:(\d+), LANES=8:(\d+) cycles", simulation)
assert m, "Missing completed functional/latency verification"
rows = []
for index, lanes in enumerate([1, 2, 8]):
    path = root / f"results/lanes{lanes}.json"
    cells = json.loads(path.read_text())["modules"]["qmac8_unit"]["cells"]
    kinds = Counter(cell["type"] for cell in cells.values())
    seq = sum(count for kind, count in kinds.items() if "DFF" in kind)
    synthesis = (root / f"results/synthesis-lanes{lanes}.log").read_text()
    multipliers = re.findall(r"^\s*(\d+)\s+\$mul\s*$", synthesis, re.M)
    assert multipliers and int(multipliers[0]) == lanes
    row = dict(lanes=lanes, arithmetic_multipliers_before_mapping=int(multipliers[0]),
               tested_vectors=int(m[1]), verified_latency_cycles=int(m[index+2]),
               generic_combinational_cells=len(cells)-seq,
               generic_sequential_cells=seq, total_generic_cells=len(cells))
    rows.append(row)
    print(f"LANES={lanes}: latency={row['verified_latency_cycles']} cycles; "
          f"generic cells={len(cells)} (comb={len(cells)-seq}, sequential={seq})")
(root / "results/summary.json").write_text(json.dumps(dict(
    cases_seed=6455, designs=rows,
    limitations=["Standalone units, not integrated into Spike/CV-Wally",
                 "Generic cell counts, not SAED area",
                 "Clock period and CPU speedup not measured"]), indent=2)+"\n")
