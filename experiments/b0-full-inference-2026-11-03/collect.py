#!/usr/bin/env python3
"""Build the comparison from saved measurements, without inferring cycles."""
import hashlib,json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[2]
R=W
here=Path(__file__).resolve().parent
saved='--saved' in sys.argv
if saved:
    sources=[R/'results/incremental-inference-2026-11-03/b0.json',R/'results/full-inference-2026-11-03/scalar.json',R/'results/incremental-inference-2026-11-03/x1.json',R/'results/full-inference-2026-11-03/cpu-design.json',R/'results/full-inference-2026-11-03/tpu.json']
else:
    sources=[here/'build/scalar-b0/result.json',R/'experiments/full-inference-baseline-2026-11-03/build/scalar-b4-inline/result.json',R/'experiments/x1-qmac-inference-2026-11-03/build/x1-qmac/result.json',R/'experiments/tpu-evaluation-2026-11-02/build/qmac-inline/result.json',R/'experiments/tpu-evaluation-2026-11-02/build/tpu-inline-hoist/result.json']
for p in sources:
    if not p.exists(): raise SystemExit(f'Missing {p}. Run python3 scripts/project.py checkpoint, or pass --saved to inspect the committed results.')
rows=[json.loads(p.read_text()) for p in sources]
stock_pass=rows[0].get('stock_spike')=='PASS'
assert stock_pass or '--allow-pending' in sys.argv
assert len({x['input_sha256'] for x in rows})==1
assert len({x['model_sha256'] for x in rows})==1
assert all(x['correctness']['nodes_checked']==14 and x['correctness']['outputs_checked']==1000 and x['correctness']['failure_stage']==0 for x in rows)
names=['B0','B1','X1','X2','TPU'];nums=[x['retired'] for x in rows]
descriptions=['Scalar B=1, software requant, separate GELU LUT; no Linear inline','Scalar B=4, fused GELU LUT, Linear inline','B1 + qmac8; software requant','X1 + requant8','Tile dot command + requant8; CPU parameter reuse + inline']
payload={'scope':'Complete inference, same input/model; includes 14 stage hashes','input_sha256':rows[0]['input_sha256'],'model_sha256':rows[0]['model_sha256'],'results':[{'stage':n,'definition':d,'retired':v,'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,d,v,p in zip(names,descriptions,nums,sources)],'limitations':['B0 GELU LUT is shared for equal functionality; separate function inside same custom node, not a separate TFLM node','One synthetic input, no dataset accuracy','Spike CPU retired instructions are not hardware cycles or speedup','TPU engine has no integrated CV-Wally RTL']}
(here/'build').mkdir(exist_ok=True)
(here/'build/comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
lines=['# Full-inference incremental comparison','','Same 12-block DeiT-Tiny model, quantization, input and 14 stage checks.','GCC 15.2.0 -O2, RV64IMC, functional Spike.','','| Stage | Configuration | Retired instructions | Change from previous |','|---|---|---:|---:|']
for i,(n,d,v) in enumerate(zip(names,descriptions,nums)):
 change='—' if i==0 else f'{100*(1-v/nums[i-1]):.2f}% fewer'
 lines.append(f'| {n} | {d} | {v:,} | {change} |')
lines += ['',f'All five runs use input SHA-256 `{rows[0]["input_sha256"]}` and model SHA-256 `{rows[0]["model_sha256"]}`.','All pass 14 stage hashes and all 1,000 final logits. '+('B0 passes stock Spike.' if stock_pass else 'B0 stock-Spike confirmation is running; B0 ELF disassembly has no custom opcodes.'),'','B0 uses a noinline standalone GELU LUT function and an intermediate tensor inside the existing custom block node. It does not add a separate TFLM operator dispatch. Its W8A8 GELU still uses the same lookup table; a non-LUT baseline requires a separate implementation and correctness evaluation. B1 bundles blocking, fusion and inline, so B0→B1 is their combined effect, not an isolated measurement. X1→X2 also includes compiler/code-generation changes. TPU changes the execution engine and adds CPU parameter reuse; it is a functional Spike experiment.','','Reproduce all stages after setup with `python3 scripts/project.py checkpoint`. To inspect committed results without rerunning, use `python3 experiments/b0-full-inference-2026-11-03/collect.py --saved`. Source overlays, compiler flags and full logs are saved in the respective build directories. Original project sources are not overwritten.','','These are CPU retired-instruction counts, not CV-Wally cycles or hardware speedups. Dataset accuracy, hardware timing and physical area remain unmeasured.']
(here/'build/REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:12]))
