#!/usr/bin/env python3
"""Controlled full-model software ablations over an immutable course repository."""
import argparse, hashlib, importlib.util, json, re, shutil, subprocess, sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
DEFAULT_REPO=HERE.parents[1]
VARIANTS={'qmac-control':(False,False,False),'qmac-inline':(False,True,False),
          'tpu-control':(True,False,False),'tpu-inline':(True,True,False),
          'tpu-hoist':(True,False,True),'tpu-inline-hoist':(True,True,True)}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,log,timeout=900,input=None):
    result=subprocess.run(list(map(str,cmd)),input=input,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    log.write_text('COMMAND '+json.dumps(list(map(str,cmd)))+'\n'+result.stdout)
    if result.returncode:raise RuntimeError(str(log)+'\n'+result.stdout[-4000:])
    return result.stdout

def kernel(source,inline,hoist):
    if inline:source=source.replace('template<int B> __attribute__((noinline)) void linear(',
        'template<int B> __attribute__((always_inline)) inline void linear(')
    if hoist:
        start=source.index('    for(int r=0;r<rows;++r) for(int j=0;j<cols;++j) {')
        end=source.index('\n  }\n  return;',start)
        source=source[:start]+'''    for(int j=0;j<cols;++j) {
      const uint32_t bv=uint32_t(bias[c+j]);
      const int32_t multiplier=mult[c+j];
      const int scale_shift=shift[c+j];
#ifdef __riscv
      const uint64_t params=requant8_params(multiplier,scale_shift,0);
#endif
      for(int r=0;r<rows;++r) {
        const uint32_t bits=uint32_t(sums[r*cols+j])+bv;
#ifdef __riscv
        int8_t q=requant8_int8(low32(bits),params);
#else
        int8_t q=requant(low32(bits),multiplier,scale_shift);
#endif
        output[(row+r)*n+c+j]=lut ? lut[int(q)+128] : q;
      }
    }'''+source[end:]
    return source

def classify(folder,elf,base,counters,symbols):
    hs=[]
    for name,count in zip(['before','after'],counters):
        log=run(base+['-g',f'--instructions={count-1}',elf],folder/(name+'.log'))
        h={int(pc,16):int(n) for pc,n in re.findall(r'^([0-9a-f]+) ([0-9]+)$',log,re.M)}
        assert sum(h.values())==count
        hs.append(h)
    assert hs[0].get(symbols['inference_begin'],0)==0 and hs[1].get(symbols['inference_begin'],0)==1
    assert hs[1].get(symbols['inference_end'],0)==0
    hist=Counter(hs[1]);hist.subtract(hs[0]);assert all(n>=0 for n in hist.values())
    asm=run(['riscv64-elf-objdump','-d','-M','no-aliases',elf],folder/'disassembly.log')
    insns={int(m[1],16):(int(m[2],16),m[3]) for line in asm.splitlines() if (m:=re.match(r'^\s*([0-9a-f]+):\s+([0-9a-f]{4,16})\s+(\S+)',line))}
    mix=Counter();mn=Counter();read=write=0
    for pc,n in hist.items():
        if not n:continue
        word,m=insns[pc];op=m.removeprefix('c.').removesuffix('sp')
        if word&0xfe00707f==0x4000000b:c=m='qmac8'
        elif word&0xfe00707f==0x4200000b:c=m='requant8'
        elif word&0xfff07fff==0x4400000b:c=m='tpu_tile'
        elif op in ['lb','lbu','lh','lhu','lw','lwu','ld']:
            c='load';read+=n*dict(lb=1,lbu=1,lh=2,lhu=2,lw=4,lwu=4,ld=8)[op]
        elif op in ['sb','sh','sw','sd']:
            c='store';write+=n*dict(sb=1,sh=2,sw=4,sd=8)[op]
        elif m in ['beq','bne','blt','bge','bltu','bgeu','c.beqz','c.bnez']:c='conditional_branch'
        elif m in ['jal','jalr','c.j','c.jal','c.jr','c.jalr']:c='jump'
        elif m.startswith('mul'):c='multiply'
        elif m.startswith(('div','rem')):c='divide_remainder'
        elif m.startswith(('f','c.f','amo','lr.','sc.','v')):raise AssertionError(m)
        else:c='integer_other'
        mix[c]+=n;mn[m]+=n
    assert sum(mix.values())==counters[1]-counters[0]
    return dict(mix=dict(mix),mnemonics=dict(mn),core_read_bytes=read,core_write_bytes=write)

def evaluate(repo,out,name,full):
    tpu,inline,hoist=VARIANTS[name];folder=out/name;folder.mkdir(parents=True,exist_ok=True)
    root=folder/'src';e=root/'experiments';f=e/'deit-full-int8';t=e/'tflm-linear'
    for sub in ['deit-full-int8','tflm-linear','tpu-spike','qmac8-requant-spike']:
        dest=e/sub;dest.mkdir(parents=True,exist_ok=True)
        for src in (repo/'experiments'/sub).iterdir():
            if src.is_file() and src.suffix in ['.cc','.h','.S','.ld']:shutil.copy2(src,dest/src.name)
    frozen_path=repo/'experiments/tpu-evaluation-2026-11-02/frozen_linear_kernel.h'
    frozen_sha=sha(frozen_path)
    if frozen_sha!='ea1c1d0fe3bf776e2bc9d3cf0ad0b0388d990ed3cee48b3e54a56ad5b2742fa3':
        raise ValueError('Frozen evaluation kernel checksum mismatch')
    frozen=frozen_path.read_text()
    (t/'linear_kernel.h').write_text(kernel(frozen,inline,hoist))
    spec=importlib.util.spec_from_file_location('base',repo/'experiments/tflm-linear/build.py')
    lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
    model=repo/'experiments/deit-full-int8/build';modelsha=sha(model/'model.tflite')
    compiler=repo/'experiments/tflm-linear/vendor/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-g++'
    metadata=dict(variant=name,tpu=tpu,linear_always_inline=inline,hoist=hoist,
        source_repo_commit='a2bee01f8345082c4749393c0a2de8e532db1a3d',frozen_kernel_sha256=frozen_sha,
        model_sha256=modelsha,input_sha256=sha(model/'input.bin'),kernel_sha256=sha(t/'linear_kernel.h'))
    for target in ['host','riscv']:
        flags=lib.COMMON+(['-DUSE_TPU'] if tpu else [])
        if target=='host':cc='clang++';flags+=['-fsanitize=undefined','-fno-sanitize-recover=all']
        else:cc=compiler;flags+=['-march=rv64imc','-mabi=lp64','-mcmodel=medany','-fno-use-cxa-atexit']
        dest=folder/target;dest.mkdir(exist_ok=True);objects=[]
        for source in [f/'main.cc',f/'model.cc',t/'platform.cc']:
            obj=dest/(source.name+'.o');objects.append(obj)
            run([cc]+flags+lib.INCLUDES+['-I'+str(model),'-DMODEL_FILE="'+str(model/'model.tflite')+'"',
                '-DINPUT_FILE="'+str(model/'input.bin')+'"','-c',source,'-o',obj],dest/(source.name+'.log'))
        library=repo/'experiments/tflm-linear/build'/target/'libtflm.a'
        command=[cc]+flags+objects+[library]
        if target=='riscv':
            for source in [t/'start.S',model/'model.S']:
                obj=dest/(source.name+'.o');run([cc]+flags+['-c',source,'-o',obj],dest/(source.name+'.log'));command.append(obj)
            command+=['-nostartfiles','-Wl,--gc-sections,--no-relax,-Map='+str(dest/'program.map'),'-T',t/'link.ld','-lm','-lc','-lgcc']
        else:command+=['-Wl,-dead_strip']
        exe=dest/('program.elf' if target=='riscv' else 'program')
        run(command+['-o',exe],dest/'link.log')
        if target=='host':
            log=run([exe],dest/'result.log');assert '1000 exact logits' in log;metadata['host_ubsan']='PASS'
        else:
            metadata['flags']=flags;metadata['compiler']=run([cc,'--version'],dest/'compiler.log').splitlines()[0]
            metadata['elf_sha256']=sha(exe)
    elf=folder/'riscv/program.elf'
    nm=run(['riscv64-elf-nm',elf],folder/'symbols.log')
    symbols={p[2]:int(p[0],16) for line in nm.splitlines() if len(p:=line.split())==3}
    base=['spike','--isa=RV64IMC','-m32','--extlib='+str(repo/'experiments/tflm-linear/build/libqmac8.dylib'),'--extension=xqmac8']
    if tpu:base+=['--extlib='+str(repo/'experiments/tpu-spike/build/libtpu.dylib'),'--extension=xtpu']
    fields=['nodes_checked','blocks_checked','outputs_checked','arena_used','top1','failure_stage','failed_node']
    cmds=f'until pc 0 {symbols["inference_begin"]:x}\nreg 0 minstret\nuntil pc 0 {symbols["inference_end"]:x}\nreg 0 minstret\nuntil pc 0 {symbols["test_finished"]:x}\nreg 0 a0\n'
    cmds+=''.join(f'mem 0 {symbols[x]:x}\n' for x in fields)+'rs\n';script=folder/'correctness.cmd';script.write_text(cmds)
    log=run(base+['-d','--instructions=10000000000','--debug-cmd='+str(script),elf],folder/'correctness.log')
    values=[int(x,16) for x in re.findall(r'^0x([0-9a-f]+)$',log,re.M)]
    assert len(values)==3+len(fields),log
    state={key:v&0xffffffff for key,v in zip(fields,values[3:])}
    assert values[2]==0 and [state[k] for k in ['nodes_checked','blocks_checked','outputs_checked','failure_stage','failed_node']]==[14,12,1000,0,0],state
    metadata.update(retired=values[1]-values[0],correctness=state)
    sizes=run(['riscv64-elf-size','-A',elf],folder/'size.log')
    metadata['sections']={m[1]:int(m[2]) for line in sizes.splitlines() if (m:=re.match(r'^(\.\S+)\s+(\d+)\s+',line))}
    if tpu:
        metadata['accelerator']=json.loads(re.search(r'TPU_STATS (\{[^\n]+\})',log)[1])
        assert metadata['accelerator']['macs']==1255044864
    if full:metadata.update(classify(folder,elf,base,values[:2],symbols))
    (folder/'result.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(name,'retired',metadata['retired'],'text',metadata['sections']['.text'],flush=True)
    return metadata

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=DEFAULT_REPO);p.add_argument('--out',type=Path,default=HERE/'build');p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS));p.add_argument('--histograms',action='store_true');a=p.parse_args()
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    results=[evaluate(a.repo.resolve(),a.out,v,a.histograms) for v in a.variants]
    (a.out/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
