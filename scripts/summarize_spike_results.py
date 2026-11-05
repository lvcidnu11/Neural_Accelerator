#!/usr/bin/env python3
"""Build/check a Spike-only results overview from committed measurements.
No simulator is run and no measured count is replaced by an estimate.
"""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
 provenance={}
 def read(name):
  path=ROOT/name;raw=path.read_bytes();provenance[name]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
 ref=read('results/reproduced/full-inference.json');macs=ref['mathematical_macs'];assert macs==1253683200
 inputs=[('B0','Scalar B=1; separate GELU LUT; software requant; no inline','results/incremental-inference-2026-11-03/b0.json'),('B1','Scalar B=4; fused GELU LUT; Linear inline','results/full-inference-2026-11-03/scalar.json'),('X1','B1 + qmac8; software requant','results/incremental-inference-2026-11-03/x1.json'),('X2','X1 + requant8','results/full-inference-2026-11-03/cpu-design.json'),('TPU','Tile offload + requant8; parameter reuse + inline','results/full-inference-2026-11-03/tpu.json')]
 rows=[];models=set();images=set()
 for stage,definition,source in inputs:
  r=read(source);c=r['correctness'];assert c['nodes_checked']==14 and c['blocks_checked']==12 and c['outputs_checked']==1000 and c['failure_stage']==0
  assert r['host_ubsan']=='PASS'
  if 'mix' in r:assert sum(r['mix'].values())==r['retired']
  models.add(r['model_sha256']);images.add(r['input_sha256'])
  if stage=='B0':assert r['stock_spike']=='PASS'
  retired=r['retired'];rows.append(dict(stage=stage,definition=definition,source=source,retired_instructions=retired,mathematical_macs=macs,macs_per_retired_instruction=macs/retired,instruction_count_ratio_vs_B0=rows[0]['retired_instructions']/retired if rows else 1.0,reduction_vs_previous=1-retired/rows[-1]['retired_instructions'] if rows else None,text_bytes=r['sections']['.text']))
 assert len(models)==len(images)==1 and ref['model_sha256'] in models
 fc=read('results/reproduced/fc1.json')['197x768'];fc_macs=197*192*768
 fcrows=[dict(variant=k,definition=fc[k]['configuration'],retired_instructions=fc[k]['invoke_retired'],mathematical_macs=fc_macs,macs_per_retired_instruction=fc_macs/fc[k]['invoke_retired']) for k in ['baseline','qmac_only','custom']]
 block=[r for r in read('results/reproduced/blocking-fusion.json') if r['scope'].startswith('Full fc1')]
 blockrows=[dict(blocking_factor=r['blocking_factor'],fused=r['fused'],retired_instructions=r['invoke_retired'],macs_per_retired_instruction=fc_macs/r['invoke_retired'],arena_used_bytes=r['correctness']['arena_used']) for r in block]
 sw=read('results/software-2026-11-02/comparison.json')['variants']
 for r in sw:
  assert r['model_sha256'] in models and r['input_sha256'] in images and r['host_ubsan']=='PASS'
  assert r['correctness']['nodes_checked']==14 and r['correctness']['outputs_checked']==1000 and r['correctness']['failure_stage']==0
 software=[dict(variant=r['variant'],retired_instructions=r['retired'],macs_per_retired_instruction=macs/r['retired'],text_bytes=r['sections']['.text']) for r in sw]
 assert software[1]['retired_instructions']==rows[3]['retired_instructions'] and software[-1]['retired_instructions']==rows[-1]['retired_instructions']
 tpu=read(inputs[-1][2]);acc=tpu['accelerator'];assert acc['macs']==ref['padded_macs']
 payload=dict(updated='2026-11-06',measurement_scope='Functional Spike CPU retired instructions; saved measurements, not a new full-model rerun',model_sha256=next(iter(models)),input_sha256=next(iter(images)),full_inference=rows,fc1=fcrows,fc1_blocking_fusion=blockrows,full_inference_software_ablation=software,tpu_engine=dict(commands=acc['commands'],padded_macs=acc['macs'],logical_read_bytes=acc['read_bytes'],logical_write_bytes=acc['write_bytes']),source_sha256=provenance,limitations=['Single synthetic input; no dataset accuracy claim','Instruction-count ratios are not cycle or wall-clock speedups','MAC/instruction uses useful mathematical model MACs divided by CPU retired instructions; TPU work is offloaded','Full-inference, FC1-only, and blocking/fusion studies have distinct scopes; ratios must not be combined','B0 to B1 combines blocking, fusion and inline; final TPU combines tile offload, parameter reuse and inline','No CV-Wally or RTL measurements are included'])
 lines=['# Spike results overview','','Updated 2026-11-06 from committed measurements (2026-10-25 to 2026-11-03).','This update recalculates derived metrics and documentation; it does not claim a new full-model run.','','## Complete DeiT-Tiny: primary comparison','','**B0 is 9,115,503,491 instructions (9.116 billion). The final TPU path is 214,329,448 (0.214 billion).**','B1 at 6.096 billion is the software-optimized scalar stage, not the unoptimized B0 baseline.','The earlier 862-million qmac and 262-million TPU counts belong to pre-inline/pre-hoist variants below.','','Every row executes the same complete 12-block model and the same synthetic input.','Each checks 14 stage hashes and all 1,000 final integer logits.','The measured interval includes operator dispatch and the stage-hash checks.','','| Stage | Configuration | CPU retired instructions | Useful MAC / CPU instruction | Reduction from previous | Count ratio vs B0 |','|---|---|---:|---:|---:|---:|']
 for r in rows:
  change='—' if r['reduction_vs_previous'] is None else f"{100*r['reduction_vs_previous']:.2f}%"
  lines.append(f"| {r['stage']} | {r['definition']} | {r['retired_instructions']:,} | {r['macs_per_retired_instruction']:.4f} | {change} | {r['instruction_count_ratio_vs_B0']:.2f}x |")
 lines+=['',f'The full model contains **{macs:,} useful mathematical MACs**.','MAC/instruction is this fixed numerator divided by each row\'s CPU retired count.','For TPU it includes accelerator work, so it is not the number of MACs executed directly by the CPU per instruction.','One tile command hides many operations and memory accesses from CPU retirement.','The final count ratio is 42.53x versus B0; this is **not a measured hardware speedup**.','','B0 uses the same INT8 GELU LUT as the optimized paths, in a separate noinline function inside','the existing block node. B0 is not upstream TFLM full-model inference or a non-LUT GELU implementation.','B0 to B1 combines B=4 blocking, FC/GELU fusion and inline. X1 to X2 also includes compiler code-generation effects.','The final TPU row includes the CPU parameter-reuse and inline improvements.','','## Full-model software ablations','','These full-inference variants use the same mathematical MAC numerator and input/reference.','Compare each inline/hoist variant against its own original execution path.','','| Configuration | CPU retired instructions | Useful MAC / CPU instruction | ELF .text bytes |','|---|---:|---:|---:|']
 for r in software:lines.append(f"| {r['variant']} | {r['retired_instructions']:,} | {r['macs_per_retired_instruction']:.4f} | {r['text_bytes']:,} |")
 lines+=['','The original qmac path is 862,447,442; inline reduces it to the X2 value of 830,264,926.','The original TPU path is 262,234,932; hoist + inline reduces it to the final 214,329,448.','These are supporting ablations, not replacement B0 baselines.','','## FC1-only instruction experiments','','FC1 has M=197, K=192, N=768: **29,048,832 MACs**. The interval is TFLM Invoke().','These counts cover a single operator, not the complete model.','','| FC1 implementation | CPU retired instructions | MAC / CPU instruction |','|---|---:|---:|']
 for r in fcrows:lines.append(f"| {r['definition']} | {r['retired_instructions']:,} | {r['macs_per_retired_instruction']:.4f} |")
 lines+=['','For the next study, every row uses the same shared qmac/requant kernel and includes the same GELU LUT.','Its B=1 row is a packed-kernel specialization, **not scalar stage B1**.','','| Channels per block | GELU fusion | CPU retired instructions | MAC / CPU instruction | Arena bytes |','|---|---|---:|---:|---:|']
 for r in blockrows:lines.append(f"| {r['blocking_factor']} | {'fused' if r['fused'] else 'separate'} | {r['retired_instructions']:,} | {r['macs_per_retired_instruction']:.4f} | {r['arena_used_bytes']:,} |")
 lines+=['','Do not multiply the FC1, blocking/fusion and full-model count ratios together.','','## TPU traffic and correctness boundaries','','The final Spike TPU variant executes **95,225 tile commands** and **1,255,044,864 padded products**.','Its engine performs 322,601,216 logical read bytes and 23,898,544 logical write bytes,','reported separately from CPU loads/stores. The padded product count is not the useful-MAC numerator above.','Logical bytes are not cache/DRAM traffic. Analytical engine timing counters are not included as measurements.','','All five full-model paths match the same project integer reference. This establishes tested arithmetic','consistency, not calibrated dataset accuracy. One synthetic image and synthetic activation scales','cannot establish ImageNet top-1/top-5 accuracy. No hardware cycles, Fmax or physical area are reported here.','','## Evidence and reproduction','','- [Machine-readable summary](../results/spike-summary.json) contains relative evidence paths and SHA-256 hashes.','- [Five-stage source measurements](../results/incremental-inference-2026-11-03/) and [B1/X2/TPU profiles](../results/full-inference-2026-11-03/).','- [Software ablations](../results/software-2026-11-02/comparison.json), [FC1](../results/reproduced/fc1.json), [blocking/fusion](../results/reproduced/blocking-fusion.json).','','Regenerate/check the summary from committed data:','','```sh','python3 scripts/summarize_spike_results.py','python3 scripts/summarize_spike_results.py --check','```','','Rerun the actual measurements after installing the pinned dependencies:','','```sh','python3 scripts/project.py setup','python3 scripts/project.py checkpoint','```','','The summary command does not execute Spike. The checkpoint command executes the workloads;','fresh counts must be verified against reference outputs and their tool/source configuration.']
 table=['| Stage | Configuration | CPU retired instructions | Useful MAC / CPU instruction | Count ratio vs B0 |','|---|---|---:|---:|---:|']
 for r in rows:table.append(f"| {r['stage']} | {r['definition']} | {r['retired_instructions']:,} | {r['macs_per_retired_instruction']:.4f} | {r['instruction_count_ratio_vs_B0']:.2f}x |")
 start='<!-- SPIKE_RESULTS_START -->';end='<!-- SPIKE_RESULTS_END -->'
 readme=ROOT/'README.md';body=readme.read_text();assert start in body and end in body
 block=start+'\n\n'+'\n'.join(table)+'\n\n'+end
 expected_readme=body[:body.index(start)]+block+body[body.index(end)+len(end):]
 if a.check:assert body==expected_readme,'Stale README results table'
 else:readme.write_text(expected_readme)
 outputs={ROOT/'results/spike-summary.json':json.dumps(payload,indent=2)+'\n',ROOT/'docs/spike-results.md':'\n'.join(lines)+'\n'}
 for path,text in outputs.items():
  if a.check:
   assert path.read_text()==text,f'Stale generated summary: {path}'
  else:path.write_text(text)
 print('PASS Spike summary: 5 matched full-model stages; 6 software ablations; 3 FC1 variants; 6 blocking/fusion variants; source hashes verified.')
if __name__=='__main__':main()
