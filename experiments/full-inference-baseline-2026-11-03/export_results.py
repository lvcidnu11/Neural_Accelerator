import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
out=HERE/'results';out.mkdir(exist_ok=True)
scalar=json.loads((HERE/'build/scalar-b4-inline/result.json').read_text())
assert scalar['stock_spike']=='PASS'
cpu=json.loads((REPO/'results/software-2026-11-02/qmac-inline.json').read_text())
tpu=json.loads((REPO/'results/software-2026-11-02/tpu-inline-hoist.json').read_text())
assert scalar['model_sha256']==cpu['model_sha256']==tpu['model_sha256']
assert scalar['input_sha256']==cpu['input_sha256']==tpu['input_sha256']
for n,d in [('scalar',scalar),('cpu-design',cpu),('tpu',tpu)]:
 assert [d['correctness'][x] for x in ['nodes_checked','blocks_checked','outputs_checked','failure_stage','failed_node']]==[14,12,1000,0,0]
 (out/(n+'.json')).write_text(json.dumps(d,indent=2)+'\n')
b,d,t=[x['retired'] for x in [scalar,cpu,tpu]]
summary=dict(date='2026-11-03',scope='Complete DeiT-Tiny Invoke including 14 diagnostic stage hashes, same input and quantization',
 baseline='Software-optimized scalar RV64IMC, B=4/fused GELU LUT/Linear inline, scalar MAC/software requant',
 configuration_instructions=dict(scalar=b,cpu_design=d,tpu=t),
 reductions=dict(cpu_vs_scalar=1-d/b,tpu_vs_cpu=1-t/d,tpu_vs_scalar=1-t/b),
 instruction_count_ratios=dict(scalar_over_cpu=b/d,scalar_over_tpu=b/t,cpu_over_tpu=d/t),
 scalar_kernel_tests=json.loads((HERE/'build/scalar-b4-inline/kernel-tests.json').read_text()),
 canonical_integration=json.loads((HERE/'integration.json').read_text()) if (HERE/'integration.json').exists() else None,
 limitations=['CPU instruction counts are not measured hardware speedup','One synthetic input, no representative calibration/accuracy','No integrated RTL cycles/CPI/Fmax/SAED area'])
(out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
files=[HERE/n for n in ['run.py','kernel_checks.py','export_results.py','REPORT.md']]
files += [HERE/'build/scalar-b4-inline'/n for n in ['result.json','correctness.log','stock-spike.log','before.log','after.log','disassembly.log','riscv/program.elf','src/experiments/tflm-linear/linear_kernel.h']]
(out/'manifest.json').write_text(json.dumps({str(f.relative_to(HERE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(summary['reductions'],indent=2))
