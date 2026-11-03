#!/usr/bin/env python3
"""Full model B0: scalar B=1, separate GELU LUT, no Linear inlining."""
import importlib.util,json,subprocess,sys,re,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
WS=HERE.parents[1]
REPO=WS
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
scalar=load('scalar',WS/'experiments/full-inference-baseline-2026-11-03/run.py')
base=scalar.base
base.VARIANTS['scalar-b0']=(False,False,False)
out=HERE/'build';folder=out/'scalar-b0'
def kernel(source,inline,hoist):
 source=scalar.scalar_kernel(source,False,False)
 p=folder/'src/experiments/deit-full-int8/model.cc';s=p.read_text()
 assert s.count('project::linear<4>')==3
 s=s.replace('project::linear<4>','project::linear<1>')
 needle='alignas(16) int8_t queries[T*64],keys[T*64],values[64*PAD],scores[T*T],probs[T*PAD],head_output[T*64];'
 assert s.count(needle)==1
 s=s.replace(needle,needle+'\nalignas(16) int8_t pre_gelu[T*H];')
 needle='void block(const int8_t* x,int8_t* y,Params p) {'
 assert s.count(needle)==1
 s=s.replace(needle,'__attribute__((noinline)) void gelu_separate(const int8_t* in,int8_t* out,int count,const int8_t* lut) {\n  for(int i=0;i<count;++i) out[i]=lut[int(in[i])+128];\n}\n'+needle)
 needle='  dense(normalized,hidden,p,44,T,D,H,p.get<int8_t>(48));'
 assert s.count(needle)==1
 s=s.replace(needle,'  dense(normalized,pre_gelu,p,44,T,D,H);\n  gelu_separate(pre_gelu,hidden,T*H,p.get<int8_t>(48));')
 p.write_text(s)
 return source
original=base.run
def run(cmd,log,timeout=900,input=None):
 cmd=[('--instructions=80000000000' if str(c)=='--instructions=10000000000' else c) for c in cmd]
 return original(cmd,log,timeout=3600,input=input)
def main():
    base.kernel=kernel
    base.run=run
    out.mkdir(exist_ok=True)
    result=base.evaluate(REPO,out,'scalar-b0',False)
    result['baseline_definition']='B0 project scalar: one output at a time, software requantization, no Linear inline, separate noinline GELU LUT traversal and intermediate tensor. Same full 14-node graph, no separate TFLM GELU node.'
    elf=folder/'riscv/program.elf'
    basecmd=['spike','--isa=RV64IMC','-m32','-d','--instructions=80000000000','--debug-cmd='+str(folder/'correctness.cmd'),elf]
    log=base.run(basecmd,folder/'stock-spike.log')
    values=[int(x,16) for x in re.findall(r'^0x([0-9a-f]+)$',log,re.M)]
    assert len(values)>=10 and values[2]==0 and [values[i]&0xffffffff for i in [3,4,5,8,9]]==[14,12,1000,0,0],values
    assert values[1]-values[0]==result['retired']
    result['stock_spike']='PASS';result['model_cc_sha256']=hashlib.sha256((folder/'src/experiments/deit-full-int8/model.cc').read_bytes()).hexdigest()
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS B0',result['retired'],flush=True)

if __name__=="__main__":
    main()
