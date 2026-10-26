// Standalone Spike extension compiled against the installed Spike C++ headers.
// Xqmac8 v0.1: custom-0, funct7=0x20, funct3=0, RV64I register set only.
#include <riscv/extension.h>
#include <cstdint>

namespace {
constexpr uint32_t MATCH_QMAC8 = 0x4000000b;
constexpr uint32_t MASK_QMAC8 = 0xfe00707f;

int lane(uint64_t bits, unsigned i) {
  unsigned value = (bits >> (8*i)) & 255;
  return value < 128 ? static_cast<int>(value) : static_cast<int>(value)-256;
}

template<bool Logged>
reg_t execute(processor_t* p, insn_t insn, reg_t pc) {
  state_t* state = p->get_state();
  // Snapshot ALL sources before writeback, including overlapping rd/rs1/rs2.
  const uint64_t a = state->XPR[insn.rs1()];
  const uint64_t w = state->XPR[insn.rs2()];
  const uint32_t old_acc = static_cast<uint32_t>(state->XPR[insn.rd()]);
  int64_t dot = 0;
  for (unsigned i=0; i<8; ++i) dot += lane(a,i)*lane(w,i);
  // Unsigned arithmetic explicitly implements modulo 2^32 without signed UB.
  const uint32_t low = old_acc + static_cast<uint32_t>(dot);
  const reg_t result = (low & 0x80000000u) ?
      (UINT64_C(0xffffffff00000000) | low) : static_cast<reg_t>(low);
  if (insn.rd() != 0) {
    if constexpr (Logged) state->log_reg_write[insn.rd() << 4] = {result,0};
    state->XPR.write(insn.rd(),result);
  }
  return pc+4;
}

class register_arg final : public arg_t {
  unsigned field;
public:
  explicit register_arg(unsigned f) : field(f) {}
  std::string to_string(insn_t insn) const override {
    return xpr_name[field==0 ? insn.rd() : field==1 ? insn.rs1() : insn.rs2()];
  }
};
const register_arg rd_arg(0), rs1_arg(1), rs2_arg(2);

class qmac8_extension final : public extension_t {
public:
  const char* name() const override { return "xqmac8"; }
  std::vector<insn_desc_t> get_instructions(const processor_t&) override {
    return {{MATCH_QMAC8,MASK_QMAC8,
      &::illegal_instruction,&execute<false>,
      &::illegal_instruction,&::illegal_instruction,
      &::illegal_instruction,&execute<true>,
      &::illegal_instruction,&::illegal_instruction}};
  }
  std::vector<disasm_insn_t*> get_disasms(const processor_t* = nullptr) override {
    return {new disasm_insn_t("qmac8",MATCH_QMAC8,MASK_QMAC8,
                             {&rd_arg,&rs1_arg,&rs2_arg})};
  }
};
}
REGISTER_EXTENSION(xqmac8, [](){ return new qmac8_extension; })
