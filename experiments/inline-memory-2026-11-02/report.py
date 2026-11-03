import hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;B=HERE/'build';out=HERE/'results';out.mkdir(exist_ok=True)
names=['qmac-control','qmac-inline','tpu-control','tpu-inline','tpu-hoist','tpu-inline-hoist']
mem={n:json.loads((B/n/'memory.json').read_text()) for n in names}
cache=[json.loads(p.read_text()) for n in names for p in (B/n).glob('cache-*.json')]
pairs=[]
for a,b in [('qmac-control','qmac-inline'),('tpu-control','tpu-inline'),('tpu-hoist','tpu-inline-hoist')]:
 x,y=mem[a],mem[b];pairs.append(dict(control=a,inline=b,extra_text_bytes=y['sections']['.text']-x['sections']['.text'],
   text_growth=(y['sections']['.text']/x['sections']['.text']-1),instruction_reduction=1-y['retired']/x['retired'],
   instructions_saved=x['retired']-y['retired'],extra_observed_stack_bytes=y['observed_stack_touched_bytes']-x['observed_stack_touched_bytes']))
report=dict(scope='Full startup and model fixture for stack writes, Invoke ROI for instructions and cache counts',memory=mem,cache=cache,pairs=pairs,
    recommendation='Use parameter hoisting first on TPU. Keep optional inlining when hundreds of code bytes and measured stack margin are affordable. No RTL speedup claim.',
    caveats=['Stack sentinel is the deepest written word, not a formal worst-case bound','No stack reservation or static data increase. Code itself resides in RAM in current linker layout.',
             'Cache model is functional and uses instruction start addresses, 4 ways and 64-byte lines. It is not CV-Wally.',
             'Installed Spike byte-read counter contains address-inflated values and is excluded. Access totals equal capped retired counters.',
             'Cache ROI uses identical warmup prefix with deterministic before/after subtraction. No data-cache or cycle modeling.'])
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
for n,d in mem.items():(out/(n+'.json')).write_text(json.dumps(d,indent=2)+'\n')
(out/'icache.json').write_text(json.dumps(cache,indent=2)+'\n')
manifest={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'audit.py',HERE/'report.py']}
for name in names:
 for f in ['memory.json','stack.log','stack.elf','main.cc.su','model.cc.su','platform.cc.su']:
  p=B/name/f;manifest[str(p.relative_to(HERE))]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in B.glob('*/*-before.log'):manifest[str(p.relative_to(HERE))]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in B.glob('*/*-after.log'):manifest[str(p.relative_to(HERE))]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in [Path(shutil.which('spike')).resolve(),Path('/opt/homebrew/include/riscv/mmu.h').resolve(),Path('/opt/homebrew/include/riscv/cachesim.h').resolve()]:manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for p in pairs:print(p)
