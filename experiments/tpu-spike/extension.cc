// TPU-inspired functional tile engine; timing is modeled separately.
#include <riscv/extension.h>
#include <riscv/mmu.h>
#include <array>
#include <cstdio>
#include <cstdint>
namespace {
constexpr uint32_t MATCH=0x4400000b, MASK=0xfff07fff; // rd=rs2=0
struct Stats {
  uint64_t commands=0,macs=0,read_bytes=0,write_bytes=0,compute_cycles=0,serial_cycles=0;
  ~Stats() { std::fprintf(stderr,"TPU_STATS {\"commands\":%llu,\"macs\":%llu,\"read_bytes\":%llu,\"write_bytes\":%llu,\"compute_cycles\":%llu,\"serial_cycles_bw16\":%llu}\n",
    (unsigned long long)commands,(unsigned long long)macs,(unsigned long long)read_bytes,
    (unsigned long long)write_bytes,(unsigned long long)compute_cycles,(unsigned long long)serial_cycles); }
} stats;
template<bool Logged> reg_t execute(processor_t* p,insn_t insn,reg_t pc) {
  (void)Logged;
  auto mmu=p->get_mmu(); uint64_t d=p->get_state()->XPR[insn.rs1()];
  if(d&7) return ::illegal_instruction(p,insn,pc);
  std::array<uint64_t,8> f;
  for(unsigned i=0;i<8;++i) f[i]=mmu->load<uint64_t>(d+8*i);
  const auto [x,w,out,k,xs,ws,m,n]=f;
  if(m<1 || m>8 || n<1 || n>8 || k>1024 || xs<k || ws<k || (out&3))
    return ::illegal_instruction(p,insn,pc);
  // Stage operands once. MMU accesses preserve protection/trap behavior;
  // source operands are snapshotted before any destination write.
  std::array<int8_t,8192> a{},b{};
  for(unsigned r=0;r<m;++r) for(unsigned i=0;i<k;++i) a[r*k+i]=mmu->load<int8_t>(x+r*xs+i);
  for(unsigned c=0;c<n;++c) for(unsigned i=0;i<k;++i) b[c*k+i]=mmu->load<int8_t>(w+c*ws+i);
  for(unsigned r=0;r<m;++r) for(unsigned c=0;c<n;++c) {
    uint32_t acc=0;
    for(unsigned i=0;i<k;++i) acc+=uint32_t(int(a[r*k+i])*int(b[c*k+i]));
    mmu->store<uint32_t>(out+4*(r*n+c),acc);
  }
  ++stats.commands;stats.macs+=m*n*k;
  stats.read_bytes+=64+(m+n)*k;stats.write_bytes+=4*m*n;
  // Hypothetical output-stationary systolic schedule, not Spike time.
  const uint64_t compute=k ? k+m+n-2 : 0;
  stats.compute_cycles+=compute;
  stats.serial_cycles+=compute+(64+(m+n)*k+15)/16+(4*m*n+15)/16;
  return pc+4;
}
class TileExtension final:public extension_t {
 public:
  const char* name() const override { return "xtpu"; }
  std::vector<insn_desc_t> get_instructions(const processor_t&) override {
    return {{MATCH,MASK,&::illegal_instruction,&execute<false>,&::illegal_instruction,&::illegal_instruction,
      &::illegal_instruction,&execute<true>,&::illegal_instruction,&::illegal_instruction}};
  }
  std::vector<disasm_insn_t*> get_disasms(const processor_t* =nullptr) override {return {};}
};
}
REGISTER_EXTENSION(xtpu, [](){return new TileExtension;})
