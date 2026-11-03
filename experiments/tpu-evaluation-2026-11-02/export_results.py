import hashlib,json
from pathlib import Path
from evaluate import HERE,DEFAULT_REPO,VARIANTS
out=HERE/'results';out.mkdir(exist_ok=True)
rows=[]
for name in VARIANTS:
    d=json.loads((HERE/'build'/name/'result.json').read_text());(out/(name+'.json')).write_text(json.dumps(d,indent=2)+'\n');rows.append(d)
q,t,final=[next(r for r in rows if r['variant']==v) for v in ['qmac-inline','tpu-control','tpu-inline-hoist']]
checks={name:json.loads((HERE/'build'/name/'kernel-tests.json').read_text()) for name in ['qmac-inline','tpu-inline-hoist']}
report=dict(date='2026-11-02',scope='Same full DeiT-Tiny graph, synthetic input, 14 stage hashes, final 1000 logits',
    baseline_commit='a2bee01',variants=rows,selected_kernel_tests=checks,
    reductions=dict(qmac_inline=1-q['retired']/862447442,tpu_inline_hoist=1-final['retired']/t['retired'],
        tpu_final_vs_inline_qmac=1-final['retired']/q['retired'],tpu_final_vs_original_qmac=1-final['retired']/862447442),
    limits=['Retired instructions, not hardware cycles','No full-model scalar B0/B1 baseline','One synthetic input, uncalibrated activation scales','No new accelerator hardware, area or Fmax evidence'])
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
manifest={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/n for n in ['evaluate.py','kernel_checks.py','export_results.py']]}
for name in VARIANTS:
    for fname in ['result.json','src/experiments/tflm-linear/linear_kernel.h','riscv/program.elf','host/program']:
        p=HERE/'build'/name/fname;manifest[str(p.relative_to(HERE))]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(report['reductions'],indent=2))
