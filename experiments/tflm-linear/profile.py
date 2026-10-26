#!/usr/bin/env python3
"""Exact ROI profiling using Spike debugger CSR snapshots + PC histograms.

No target CSR instructions, and no per-instruction multi-GB commit log needed.
The two histogram runs stop at Invoke entry/exit; their difference is the ROI.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shape',default='197x768',choices=['2x8','197x768'])
    parser.add_argument('--custom',action='store_true')
    parser.add_argument('--qmac-only',action='store_true')
    parser.add_argument('--gelu',action='store_true')
    parser.add_argument('--project-kernel',action='store_true')
    parser.add_argument('--block',type=int,choices=[1,2,4],default=1)
    parser.add_argument('--fused',action='store_true')
    a=parser.parse_args()
    if a.fused: a.gelu=True
    if a.block>1 or a.fused: a.custom=True
    if a.qmac_only: a.custom=True
    shape=a.shape+('-gelu' if a.gelu else '')+('-fused' if a.fused else '')
    variant='qmac-only' if a.qmac_only else ('custom' if a.custom else 'baseline')
    if a.project_kernel: variant+='-kernel'
    if a.block>1: variant+=f'-b{a.block}'
    out=ROOT/'build'/shape/('riscv-'+variant)
    elf=out/'program.elf'
    symbols={}
    for line in subprocess.check_output(['riscv64-elf-nm',str(elf)],text=True).splitlines():
        p=line.split()
        if len(p)==3: symbols[p[2]]=int(p[0],16)
    disasm=subprocess.check_output(['riscv64-elf-objdump','-d','-M','no-aliases',str(elf)],text=True)
    (out/'program.disasm').write_text(disasm)
    instructions={}
    for line in disasm.splitlines():
        m=re.match(r'^\s*([0-9a-f]+):\s+([0-9a-f]{4,16})\s+(\S+)',line)
        if m: instructions[int(m[1],16)]=(int(m[2],16),m[3])
    base=['spike','--isa=RV64IMC','-m32','--instructions=1000000000']
    # Both controlled variants use the same extension-loaded Spike. run.py also
    # executes each baseline with stock Spike as an independent check.
    base += ['--extlib='+str(ROOT/'build/libqmac8.dylib'),'--extension=xqmac8']
    def until(label): return f'until pc 0 {symbols[label]:x}\n'
    def execute(name,commands,histogram=False):
        script=out/(name+'.cmd'); script.write_text(commands)
        command=base+(['-g'] if histogram else [])+['-d','--debug-cmd='+str(script),str(elf)]
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        (out/(name+'.log')).write_text(result.stdout)
        (out/(name+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
        if result.returncode: raise RuntimeError(result.stdout[-3000:])
        assert 'illegal' not in result.stdout.lower(),result.stdout[-3000:]
        counters=[int(m,16) for m in re.findall(r'^0x([0-9a-f]+)$',result.stdout,re.M)]
        hist={int(pc,16):int(n) for pc,n in re.findall(r'^([0-9a-f]+) ([0-9]+)$',result.stdout,re.M)}
        return counters,hist
    labels=['invoke_begin','operator_begin','operator_end']
    if a.gelu and not a.fused: labels+=['gelu_begin','gelu_end']
    labels+=['invoke_end']
    commands=''.join(until(label)+'reg 0 minstret\n' for label in labels)+'quit\n'
    post,_=execute('boundary-counters',commands)
    assert len(post)==len(labels),post
    def capped_histogram(name, counter):
        # This installed Spike's --instructions=N executes N+1 instructions.
        # Validate the exact total and marker exclusion on every run below.
        command=[x for x in base if not x.startswith('--instructions=')]+[
            '-g',f'--instructions={counter-1}',str(elf)]
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        (out/(name+'.log')).write_text(result.stdout)
        (out/(name+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
        assert result.returncode==0,result.stdout[-1000:]
        hist={int(pc,16):int(n) for pc,n in re.findall(r'^([0-9a-f]+) ([0-9]+)$',result.stdout,re.M)}
        assert sum(hist.values())==counter,(sum(hist.values()),counter)
        return hist
    h0=capped_histogram('before-invoke',post[0])
    h1=capped_histogram('after-invoke',post[-1])
    assert h0.get(symbols['invoke_begin'],0)==0
    assert h1.get(symbols['invoke_begin'],0)==1
    assert h1.get(symbols['invoke_end'],0)==0
    retired=post[-1]-post[0]
    histogram=Counter(h1)
    histogram.subtract(h0)
    assert all(n>=0 for n in histogram.values())
    assert sum(histogram.values())==retired,(sum(histogram.values()),retired)
    commands=until('test_finished')+'reg 0 a0\n'
    fields=['outputs_checked','custom_calls','arena_used','failure_stage','failed_index']
    if a.gelu: fields+=['gelu_calls','lut_entries_checked','fc_outputs_checked']
    for field in fields: commands+=f'mem 0 {symbols[field]:x}\n'
    commands+='rs\n'
    outcome,_=execute('correctness',commands)
    assert len(outcome)==len(fields)+1,outcome
    state={key:value&0xffffffff for key,value in zip(fields,outcome[1:])}
    ref=json.loads((ROOT/'build'/shape/'reference.json').read_text())
    assert outcome[0]==0 and state['failure_stage']==0 and state['outputs_checked']==ref['outputs'],state
    assert state['custom_calls']==int(a.custom),state
    if a.gelu:
        assert state['gelu_calls']==(0 if a.fused else 1) and state['lut_entries_checked']==256,state
        assert state['fc_outputs_checked']==(0 if a.fused else ref['outputs']),state
    mix=Counter(); mnemonics=Counter()
    loads={'lb','lbu','lh','lhu','lw','lwu','ld','c.lw','c.ld','c.lwsp','c.ldsp'}
    stores={'sb','sh','sw','sd','c.sw','c.sd','c.swsp','c.sdsp'}
    branches={'beq','bne','blt','bge','bltu','bgeu','c.beqz','c.bnez'}
    jumps={'jal','jalr','c.j','c.jal','c.jr','c.jalr'}
    for pc,n in histogram.items():
        if not n: continue
        word,mnemonic=instructions[pc]
        if word & 0xfe00707f==0x4000000b: category=mnemonic='qmac8'
        elif word & 0xfe00707f==0x4200000b: category=mnemonic='requant8'
        elif mnemonic in loads: category='load'
        elif mnemonic in stores: category='store'
        elif mnemonic in branches: category='conditional_branch'
        elif mnemonic in jumps: category='jump'
        elif mnemonic.startswith('mul'): category='multiply'
        elif mnemonic.startswith(('div','rem')): category='divide_remainder'
        elif mnemonic.startswith(('f','amo','lr.','sc.','v')): raise AssertionError((pc,mnemonic))
        else: category='integer_other'
        mix[category]+=n;mnemonics[mnemonic]+=n
    assert mix['qmac8']==(ref['macs']//8 if a.custom else 0),dict(mix)
    assert mix['requant8']==(ref['outputs'] if a.custom and not a.qmac_only else 0),dict(mix)
    size=subprocess.check_output(['riscv64-elf-size','-A',str(elf)],text=True)
    (out/'size.txt').write_text(size)
    result=dict(scope='Full fc1' if a.shape=='197x768' else 'Small fc1 tile',
        configuration='TFLM qmac8 + software requant' if a.qmac_only else ('TFLM qmac8 + requant8' if a.custom else 'TFLM upstream FC'),
        isa='RV64IMC',invoke_retired=retired,operator_retired=post[2]-post[1],
        instructions_per_output=retired/ref['outputs'],macs_per_instruction=ref['macs']/retired,
        mix=dict(mix),mnemonics=dict(mnemonics.most_common()),correctness=state,
        model_sha256=ref['model_sha256'],elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),
        method='Difference of PC histograms at Invoke boundaries; checked against debugger minstret delta',
        limitations=['No cycles/CPI/hardware speedup','Synthetic activations; no calibration or accuracy claim',
                     'integer_other category includes miscellaneous integer instructions'])
    if a.gelu and not a.fused:
        result.update(scope=result['scope']+' -> GELU LUT',
            gelu_retired=post[4]-post[3],gelu_instructions_per_output=(post[4]-post[3])/ref['outputs'],
            dispatch_and_other_retired=retired-(post[2]-post[1])-(post[4]-post[3]),
            gelu_lut_sha256=ref['gelu_lut_sha256'])
    result.update(blocking_factor=a.block,fused=a.fused)
    if a.fused: result.update(scope=('Full fc1' if a.shape=='197x768' else 'Small fc1 tile')+' -> GELU fused',fused_fc_gelu_retired=post[2]-post[1])
    (out/'profile.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
