#!/usr/bin/env python3
"""Complete scalar RV64IMC baseline with matched B=4/fused LUT/inlining."""
import importlib.util,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
spec=importlib.util.spec_from_file_location('evalbase',HERE.parent/'tpu-evaluation-2026-11-02/evaluate.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
ORIGINAL_KERNEL=base.kernel
SCALAR='''
#ifdef PROJECT_SCALAR_BASELINE
  // Direct scalar multiply/add, not software emulation of a packed instruction.
  // Keep B-channel input reuse and the same quantization / LUT boundary.
  for(int row=0;row<m;++row) {
    int c=0;
    for(;c+B<=n;c+=B) {
      const int8_t* xp=x+row*k;
      const int8_t* w0=w+c*k;
      uint32_t a0=uint32_t(bias[c]),a1=0,a2=0,a3=0;
      if constexpr(B>=2) a1=uint32_t(bias[c+1]);
      if constexpr(B==4){a2=uint32_t(bias[c+2]);a3=uint32_t(bias[c+3]);}
      for(int i=0;i<k;++i) {
        int v=xp[i];a0+=uint32_t(v*int(w0[i]));
        if constexpr(B>=2) a1+=uint32_t(v*int(w0[k+i]));
        if constexpr(B==4){a2+=uint32_t(v*int(w0[2*k+i]));a3+=uint32_t(v*int(w0[3*k+i]));}
      }
      output[row*n+c]=finish(a0,mult[c],shift[c],lut);
      if constexpr(B>=2) output[row*n+c+1]=finish(a1,mult[c+1],shift[c+1],lut);
      if constexpr(B==4){output[row*n+c+2]=finish(a2,mult[c+2],shift[c+2],lut);output[row*n+c+3]=finish(a3,mult[c+3],shift[c+3],lut);}
    }
    for(;c<n;++c){
      uint32_t acc=uint32_t(bias[c]);for(int i=0;i<k;++i)acc+=uint32_t(int(x[row*k+i])*int(w[c*k+i]));
      output[row*n+c]=finish(acc,mult[c],shift[c],lut);
    }
  }
  return;
#endif
'''
def scalar_kernel(source,inline,hoist):
    source=ORIGINAL_KERNEL(source,inline,False)
    source=source.replace('#ifdef __riscv','#if defined(__riscv) && !defined(PROJECT_SCALAR_BASELINE)')
    source='#define PROJECT_SCALAR_BASELINE\n'+source
    key='  static_assert(B==1 || B==2 || B==4,"supported blocking factors");'
    source=source.replace(key,key+SCALAR)
    return source

def main():
    out=HERE/'build';out.mkdir(exist_ok=True)
    base.VARIANTS['scalar-b4-inline']=(False,True,False)
    base.kernel=scalar_kernel
    # Longer instruction cap for this full scalar workload.
    original_run=base.run
    def capped_run(cmd,log,timeout=900,input=None):
        cmd=[('--instructions=50000000000' if str(c)=='--instructions=10000000000' else c) for c in cmd]
        return original_run(cmd,log,timeout=1800,input=input)
    base.run=capped_run
    d=base.evaluate(REPO,out,'scalar-b4-inline',True)
    assert not any(d['mix'].get(k,0) for k in ['qmac8','requant8','tpu_tile']),d['mix']
    d['baseline_definition']='Software-optimized scalar RV64IMC, B=4 template blocking, same fused INT8 GELU LUT and Linear inlining. Scalar MAC and software requant. Same full graph/model/input.'
    folder=out/'scalar-b4-inline'
    # Reuse the debugger script with stock Spike, omitting both custom plugins.
    log=base.run(['spike','--isa=RV64IMC','-m32','-d','--instructions=50000000000','--debug-cmd='+str(folder/'correctness.cmd'),folder/'riscv/program.elf'],folder/'stock-spike.log')
    assert log.count('0x')>=10
    import re
    values=[int(x,16) for x in re.findall(r'^0x([0-9a-f]+)$',log,re.M)]
    assert values[2]==0 and (values[3]&0xffffffff)==14 and (values[4]&0xffffffff)==12 and (values[5]&0xffffffff)==1000
    d['stock_spike']='PASS';(folder/'result.json').write_text(json.dumps(d,indent=2)+'\n')
    print('PASS: full scalar model, no custom instructions, stock Spike',d['retired'],flush=True)
if __name__=='__main__':main()
