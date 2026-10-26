# Xqmac8 v0.1: provisional RV64 custom packed MAC

Implemented and verified in Spike on 2026-10-24. Not a standard RISC-V
instruction and not yet integrated into CV-Wally. Requantization remains
ordinary software. This document freezes the current experimental arithmetic
and encoding for reproducible comparisons; future revisions must be versioned.

## Encoding

32-bit R-type, opcode custom-0=0x0b, funct3=0, funct7=0x20.

```text
31         25 24     20 19     15 14  12 11      7 6         0
funct7=0x20     rs2       rs1    000       rd     opcode=0x0b
```

Match = 0x4000000b, mask = 0xfe00707f. Reserved funct3/funct7 patterns do not
match this instruction. No load/store, CSR or implicit accumulator state.

Assembly emitted with GNU .insn, so GCC/binutils need no mnemonic patch:

```asm
.insn r 0x0b, 0, 0x20, rd, rs1, rs2
```

The plugin disassembler displays qmac8; stock objdump shows .insn/encoding.
Changing the displayed name alone cannot make an unmodified simulator execute
the instruction. The extension must be registered and loaded.

## Semantics

Snapshot the old values of rd, rs1 and rs2 before any writeback.

```text
a_i = signed_int8(old(rs1)[8*i+7 : 8*i])
w_i = signed_int8(old(rs2)[8*i+7 : 8*i])
dot = sum(a_i * w_i), i=0..7                // exact signed 19-bit range
low = (old(rd)[31:0] + dot) mod 2^32
rd = sign_extend_64(low[31:0])
PC = PC + 4
```

- Lane 0 is bits[7:0]. Ordinary little-endian 64-bit loads place the first
  memory byte in lane 0. Byte order in a packed register is explicitly defined.
- Accumulation wraps at 32 bits; it is not saturating. Output high 32 bits
  replicate bit 31. Old rd high bits do not contribute to the result.
- x0 is read as zero. If rd=x0, the computation has no architectural writeback.
- All register overlap is legal, including rd=rs1, rd=rs2, rs1=rs2 and all equal.
  Reading sources after writeback would violate this rule.
- Current descriptor supplies RV64I handlers; other XLEN/register profiles use
  illegal-instruction handlers. Only RV64IMC execution has been tested here.
- This instruction reads three values (old rd, rs1, rs2). Spike functional
  execution does not solve the hardware register-file third-read-port problem,
  forwarding/hazard logic, pipeline stalls or timing.

The selected Linear fixture proves all partial accumulators fit INT32, so
its results do not depend on wraparound. Directed ISA tests separately verify
that overflow follows the defined wrap rule.

## Instruction cost versus implementation

One instruction always retires once, even though the host C++ handler loops
over eight lanes. That host loop is a functional model, not an RTL timing model.
The previous one/two/eight-lane RTL experiments are separate candidates and
have not been integrated with this decode or with CV-Wally's pipeline.

Reference interfaces: installed riscv/extension.h, processor.h, disasm.h and
decode.h; full hashes and the simulator binary hash are in build/manifest.json.
Homebrew receipt records Spike source SCM revision
55b4658dbf574ba0b714083ec436ce2cb5be1998. The upstream C++ API is not a stable
ABI: compile against headers matching the exact executable used for the run.

Upstream interface reference:
https://github.com/riscv-software-src/riscv-isa-sim/blob/55b4658dbf574ba0b714083ec436ce2cb5be1998/riscv/extension.h
