#!/usr/bin/env python3
"""Exercise the unchanged local CV-Wally controller, not a full-core simulation."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
DEFAULT=Path(__file__).resolve().parents[2].parent / 'cvw'
def run(cmd,log):
 p=subprocess.run(list(map(str,cmd)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
 log.write_text('COMMAND '+json.dumps(list(map(str,cmd)))+'\n'+p.stdout)
 if p.returncode:raise RuntimeError(str(log)+'\n'+p.stdout[-5000:])
 return p.stdout

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--wally',type=Path,default=DEFAULT);args=parser.parse_args();w=args.wally.resolve();b=HERE/'build';b.mkdir(exist_ok=True)
 src=w/'src/ieu/controller.sv';s=src.read_text();ports=s[s.index('input  logic'):s.index('\n);')]
 declarations=[];inputs=[]
 for line in ports.splitlines():
  line=line.split('//')[0].strip().rstrip(',')
  m=re.match(r'(input|output)\s+logic\s*(\[[^\]]+\])?\s+(.+)',line)
  if not m:continue
  direction,width,names=m.groups();names=[n.strip() for n in names.split(',')]
  declarations.append('logic '+(width+' ' if width else '')+', '.join(names)+';')
  if direction=='input':inputs.extend(names)
 assert len(inputs)>15
 tb='''`timescale 1ns/1ps
module controller_probe;
import cvw::*;
localparam cvw_t P = '{default:'0, XLEN:64, AHBW:64, M_SUPPORTED:1'b1, ZMMUL_SUPPORTED:1'b1, ZCA_SUPPORTED:1'b1, ZICSR_SUPPORTED:1'b1};
'''+ '\n'.join(declarations)+'''
controller #(P) dut (.*);
task automatic check(input logic [31:0] insn,input bit expected_illegal,input string name);
 InstrD=insn;#1;
 if(IllegalBaseInstrD !== expected_illegal) $fatal(1,"%s expected illegal=%d got=%d",name,expected_illegal,IllegalBaseInstrD);
 $display("PASS %s instruction=%08x illegal=%d",name,insn,IllegalBaseInstrD);
endtask
initial begin
'''+ '\n'.join(n+"='0;" for n in inputs)+'''
reset=1;#1;clk=1;#1;clk=0;reset=0;
check(32'h00000013,0,"RV64 ADDI");
check(32'h02000033,0,"RV64 MUL");
check(32'h00003003,0,"RV64 LD");
check(32'h4005050b,1,"qmac8 custom-0");
check(32'h4205050b,1,"requant8 custom-0");
check(32'h4405000b,1,"xtpu descriptor custom-0");
$display("PASS: unchanged CV-Wally decoder rejects all three project custom opcodes");
$finish;
end
endmodule
'''
 (b/'controller_probe.sv').write_text(tb)
 commands=['verilator','--binary','--timing','--assert','-Wno-fatal','--top-module','controller_probe',
  '--Mdir',b/'obj','-o','controller_probe','-I'+str(w/'config/shared'),w/'src/cvw.sv',src,
  w/'src/generic/flop/flopenrc.sv',w/'src/generic/mux.sv',b/'controller_probe.sv']
 run(commands,b/'compile.log');result=run([b/'obj/controller_probe'],b/'result.log');print(result)
 evidence=dict(wally_revision=run(['git','-C',w,'rev-parse','HEAD'],b/'revision.log').strip(),
   scope='Unmodified CV-Wally controller component only, with minimal RV64IMC+CSR parameters. Not full CPU or accelerator integration.',
   verilator=run(['verilator','--version'],b/'verilator-version.log').strip(),
   tests=6,custom_instructions_rejected=['qmac8','requant8','xtpu'],standard_controls_accepted=['ADDI','MUL','LD'],
   sources_sha256={str(p.relative_to(w)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [src,w/'src/cvw.sv',w/'src/lsu/lsu.sv',w/'src/ebu/ebu.sv',w/'config/rv64gc/config.vh',w/'config/shared/BranchPredictorType.vh',w/'src/generic/flop/flopenrc.sv',w/'src/generic/mux.sv']},
   limitations=['No custom arithmetic or memory engine was added','No instruction retirement/trap handler/full ELF simulated','No performance/area/timing claim'])
 (b/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
if __name__=='__main__':main()
