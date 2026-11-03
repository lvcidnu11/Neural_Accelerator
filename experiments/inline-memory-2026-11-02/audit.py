#!/usr/bin/env python3
"""Audit static sections, untouched compile diagnostics, stack writes, and I-cache sensitivity."""
import argparse, hashlib, importlib.util, json, re, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
EVAL=HERE.parent/'tpu-evaluation-2026-11-02/build'
NAMES=['qmac-control','qmac-inline','tpu-control','tpu-inline','tpu-hoist','tpu-inline-hoist']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,log,timeout=900):
    cmd=list(map(str,cmd));p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    log.write_text('COMMAND '+json.dumps(cmd)+'\n'+p.stdout)
    if p.returncode:raise RuntimeError(str(log)+'\n'+p.stdout[-4000:])
    return p.stdout

def extension(repo,tpu):
    cmd=['spike','--isa=RV64IMC','-m32','--extlib='+str(repo/'experiments/tflm-linear/build/libqmac8.dylib'),'--extension=xqmac8']
    if tpu:cmd+=['--extlib='+str(repo/'experiments/tpu-spike/build/libtpu.dylib'),'--extension=xtpu']
    return cmd

def memory(repo,ev,out,name):
    original=ev/name;d=json.loads((original/'result.json').read_text());o=out/name;o.mkdir(parents=True,exist_ok=True)
    rows=[];objects={}
    for src in ['main.cc','model.cc','platform.cc']:
        cmd=json.loads((original/'riscv'/(src+'.log')).read_text().splitlines()[0].removeprefix('COMMAND '))
        obj=o/(src+'.o');cmd[cmd.index('-o')+1]=str(obj);cmd.insert(1,'-fstack-usage')
        run(cmd,o/(src+'.compile.log'))
        objects[src]=dict(byte_identical=sha(obj)==sha(original/'riscv'/(src+'.o')),sha256=sha(obj))
        assert objects[src]['byte_identical'],src
        for line in obj.with_suffix('.su').read_text().splitlines():
            fn,size,kind=line.rsplit('\t',2);rows.append(dict(function=fn,bytes=int(size),kind=kind))
    relevant=[r for r in rows if any(x in r['function'] for x in ['linear(','Embed','Block','Head','main()'])]
    # Paint the full 64 KiB stack reservation using registers before any C call.
    # Scan it after main returns with a register-only assembly loop.
    start=(repo/'experiments/tflm-linear/start.S').read_text()
    paint='''    li t0, -65536
    add t0, sp, t0
    li t2, 0xa5c39e71b462d08f
.Lpaint:
    sd t2, 0(t0)
    addi t0, t0, 8
    bltu t0, sp, .Lpaint
'''
    scan='''    la t1, _stack_top
    li t0, -65536
    add t0, t1, t0
    li t2, 0xa5c39e71b462d08f
.Lscan:
    beq t0, t1, .Lscan_done
    ld t3, 0(t0)
    bne t3, t2, .Lscan_done
    addi t0, t0, 8
    j .Lscan
.Lscan_done:
    sub t4, t1, t0
    la t5, stack_touched_bytes
    sd t4, 0(t5)
'''
    start=start.replace('    la sp, _stack_top\n','    la sp, _stack_top\n'+paint)
    start=start.replace('.globl test_finished\ntest_finished:',scan+'.globl test_finished\ntest_finished:')
    start+='\n.section .bss\n.balign 8\n.globl stack_touched_bytes\nstack_touched_bytes: .dword 0\n'
    (o/'start.S').write_text(start)
    cc=repo/'experiments/tflm-linear/vendor/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-g++'
    run([cc]+d['flags']+['-c',o/'start.S','-o',o/'start.o'],o/'start.log')
    link=json.loads((original/'riscv/link.log').read_text().splitlines()[0].removeprefix('COMMAND '))
    link[link.index('-o')+1]=str(o/'stack.elf')
    link=[str(o/'start.o') if x==str(original/'riscv/start.S.o') else x for x in link]
    link=[('-Wl,--gc-sections,--no-relax,-Map='+str(o/'stack.map')) if x.startswith('-Wl,') else x for x in link]
    run(link,o/'stack-link.log')
    nm=run(['riscv64-elf-nm',o/'stack.elf'],o/'symbols.log')
    symbols={p[2]:int(p[0],16) for line in nm.splitlines() if len(p:=line.split())==3}
    fields=['stack_touched_bytes','outputs_checked','nodes_checked','blocks_checked','failure_stage']
    script=f'until pc 0 {symbols["inference_begin"]:x}\nreg 0 minstret\nuntil pc 0 {symbols["inference_end"]:x}\nreg 0 minstret\nuntil pc 0 {symbols["test_finished"]:x}\nreg 0 a0\n'
    script+=''.join(f'mem 0 {symbols[x]:x}\n' for x in fields)+'rs\n';(o/'stack.cmd').write_text(script)
    log=run(extension(repo,d['tpu'])+['-d','--instructions=10000000000','--debug-cmd='+str(o/'stack.cmd'),o/'stack.elf'],o/'stack.log')
    vals=[int(x,16) for x in re.findall(r'^0x([0-9a-f]+)$',log,re.M)]
    assert len(vals)==8,vals
    assert vals[1]-vals[0]==d['retired'],(name,vals[1]-vals[0],d['retired'])
    assert vals[2]==0 and [x&0xffffffff for x in vals[4:]]==[1000,14,12,0],vals
    sections=d['sections'];report=dict(variant=name,retired=d['retired'],sections=sections,
       reserved_stack_bytes=65536,observed_stack_touched_bytes=vals[3],arena_used=d['correctness']['arena_used'],
       static_writable_bytes=sum(sections.get(s,0) for s in ['.data','.bss','.tohost']),
       loadable_payload_bytes=sum(sections.get(s,0) for s in ['.text','.rodata','.data','.tohost','.init_array']),
       diagnostics_objects=objects,compiler_stack_frames=rows,relevant_frames=relevant,
       note='Sentinel reports deepest written stack word in this complete startup/main execution, 8-byte resolution. Unwritten reserved frames may escape detection. It is not a worst-case call-stack bound.')
    (o/'memory.json').write_text(json.dumps(report,indent=2)+'\n')
    print(name,'text',sections['.text'],'stack touched',vals[3],'frames',relevant,flush=True)
    return report

def cache(repo,ev,out,name,config):
    original=ev/name;d=json.loads((original/'result.json').read_text());o=out/name;o.mkdir(exist_ok=True)
    vals=[int(x,16) for x in re.findall(r'^0x([0-9a-f]+)$',(original/'correctness.log').read_text(),re.M)]
    assert vals[1]-vals[0]==d['retired']
    results=[]
    for label,counter in zip(['before','after'],vals[:2]):
        log=run(extension(repo,d['tpu'])+['--ic='+config,'--instructions='+str(counter-1),original/'riscv/program.elf'],o/(config.replace(':','-')+'-'+label+'.log'))
        access=int(re.search(r'I\$ Read Accesses:\s+(\d+)',log)[1]);miss=int(re.search(r'I\$ Read Misses:\s+(\d+)',log)[1])
        assert access==counter,(access,counter)
        results.append((access,miss))
    access,miss=[b-a for a,b in zip(results[0],results[1])]
    r=dict(variant=name,configuration=config,roi_accesses=access,roi_misses=miss,miss_rate=miss/access,
           method='Deterministic capped runs from reset, after-minus-before inference; same warmup prefix',
           warning='Functional random-replacement model, no timing. Cache traces use instruction start addresses. Installed Spike Bytes Read counter is malformed and excluded.')
    (o/('cache-'+config.replace(':','-')+'.json')).write_text(json.dumps(r,indent=2)+'\n')
    print('CACHE',name,config,r['roi_misses'],r['miss_rate'],flush=True)
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=REPO);p.add_argument('--eval',type=Path,default=EVAL);p.add_argument('--cache-only',action='store_true');p.add_argument('--memory-only',action='store_true');a=p.parse_args()
    a.repo=a.repo.resolve();a.eval=a.eval.resolve()
    out=HERE/'build';out.mkdir(exist_ok=True)
    if not a.cache_only:
        for name in NAMES:memory(a.repo,a.eval,out,name)
    if not a.memory_only:
        for config in ['16:4:64','64:4:64']:
            for name in ['qmac-control','qmac-inline','tpu-hoist','tpu-inline-hoist']:cache(a.repo,a.eval,out,name,config)
if __name__=='__main__':main()
