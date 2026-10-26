#!/usr/bin/env python3
"""Profile complete 14-node inference, including stage hash checks."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
ROOT=Path(__file__).resolve().parent
B=ROOT/'build';OUT=B/'riscv';ELF=OUT/'program.elf'

def main():
    symbols={p[2]:int(p[0],16) for line in subprocess.check_output(['riscv64-elf-nm',str(ELF)],text=True).splitlines() if len(p:=line.split())==3}
    base=['spike','--isa=RV64IMC','-m32','--extlib='+str(ROOT.parent/'tflm-linear/build/libqmac8.dylib'),'--extension=xqmac8']
    fields=['nodes_checked','blocks_checked','outputs_checked','arena_used','top1','failure_stage','failed_node','expected_stage_hash','actual_stage_hash']
    cmd=f'until pc 0 {symbols["inference_begin"]:x}\nreg 0 minstret\nuntil pc 0 {symbols["inference_end"]:x}\nreg 0 minstret\nuntil pc 0 {symbols["test_finished"]:x}\nreg 0 a0\n'
    cmd+=''.join(f'mem 0 {symbols[field]:x}\n' for field in fields)+'rs\n'
    script=OUT/'correctness.cmd';script.write_text(cmd)
    def run(name,command):
        (OUT/(name+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
        p=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=900)
        (OUT/(name+'.log')).write_text(p.stdout)
        if p.returncode: raise RuntimeError(f'{name}: exit={p.returncode}\n{p.stdout[-3000:]}')
        assert 'illegal' not in p.stdout.lower(),p.stdout[-3000:]
        return p.stdout
    text=run('correctness',base+['-d','--instructions=10000000000','--debug-cmd='+str(script),str(ELF)])
    values=[int(v,16) for v in re.findall(r'^0x([0-9a-f]+)$',text,re.M)]
    assert len(values)==3+len(fields),values
    state={key:v&0xffffffff for key,v in zip(fields,values[3:])}
    assert values[2]==0 and state['nodes_checked']==14 and state['blocks_checked']==12 and state['outputs_checked']==1000,state
    assert state['failure_stage']==0 and state['failed_node']==0,state
    ref=json.loads((B/'reference.json').read_text());assert state['top1']==ref['quantized_top1']
    print('PASS: Spike full inference correctness',state,flush=True)
    def histogram(name,count):
        log=run(name,base+['-g',f'--instructions={count-1}',str(ELF)])
        h={int(pc,16):int(n) for pc,n in re.findall(r'^([0-9a-f]+) ([0-9]+)$',log,re.M)}
        assert sum(h.values())==count,(sum(h.values()),count)
        return h
    before=histogram('before-inference',values[0]);after=histogram('after-inference',values[1])
    assert before.get(symbols['inference_begin'],0)==0 and after.get(symbols['inference_begin'],0)==1
    assert after.get(symbols['inference_end'],0)==0
    hist=Counter(after);hist.subtract(before)
    retired=values[1]-values[0];assert sum(hist.values())==retired and all(n>=0 for n in hist.values())
    asm=subprocess.check_output(['riscv64-elf-objdump','-d','-M','no-aliases',str(ELF)],text=True);(OUT/'program.disasm').write_text(asm)
    instructions={int(m[1],16):(int(m[2],16),m[3]) for line in asm.splitlines() if (m:=re.match(r'^\s*([0-9a-f]+):\s+([0-9a-f]{4,16})\s+(\S+)',line))}
    mix=Counter();mnemonics=Counter()
    loads={'lb','lbu','lh','lhu','lw','lwu','ld','c.lw','c.ld','c.lwsp','c.ldsp'}
    stores={'sb','sh','sw','sd','c.sw','c.sd','c.swsp','c.sdsp'}
    branches={'beq','bne','blt','bge','bltu','bgeu','c.beqz','c.bnez'}
    jumps={'jal','jalr','c.j','c.jal','c.jr','c.jalr'}
    for pc,n in hist.items():
        if not n: continue
        word,m=instructions[pc]
        if word&0xfe00707f==0x4000000b: category=m='qmac8'
        elif word&0xfe00707f==0x4200000b:category=m='requant8'
        elif m in loads:category='load'
        elif m in stores:category='store'
        elif m in branches:category='conditional_branch'
        elif m in jumps:category='jump'
        elif m.startswith('mul'):category='multiply'
        elif m.startswith(('div','rem')):category='divide_remainder'
        elif m.startswith(('f','c.f','amo','lr.','sc.','v')):raise AssertionError((pc,m))
        else:category='integer_other'
        mix[category]+=n;mnemonics[m]+=n
    # 1,253,683,200 mathematical MACs, plus AV padding 197->200;
    # each QK output-channel tail (197 % 4) uses the scalar fallback.
    assert mix['qmac8']==156823872,dict(mix)
    assert mix['requant8']==5974636,dict(mix)
    size=subprocess.check_output(['riscv64-elf-size','-A',str(ELF)],text=True);(OUT/'size.txt').write_text(size)
    report=dict(scope='Complete DeiT-Tiny inference, one synthetic input; includes per-node stage-hash checks',
        isa='RV64IMC + project qmac8/requant8',retired_instructions=retired,mix=dict(mix),mnemonics=dict(mnemonics.most_common()),
        correctness=state,model_sha256=ref['model_sha256'],elf_sha256=hashlib.sha256(ELF.read_bytes()).hexdigest(),
        mathematical_macs=1253683200,padded_macs=1255044864,
        limitations=ref['limitations']+['No full-model baseline speedup or hardware cycles measured','Measured ROI includes all 14 output stage-hash checks'])
    (B/'profile.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
