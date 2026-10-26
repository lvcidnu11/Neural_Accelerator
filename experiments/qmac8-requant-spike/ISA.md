# qmac8 + requant8, experimental ISA v0.2

Implemented in an actual Spike extension on 2026-10-24. qmac8 semantics and
encoding are unchanged from ../qmac8-spike/ISA.md. Both instructions are
registered by the library under extension name xqmac8. Load ONE matching
library, not the old and new libraries together. No CV-Wally integration yet.

## requant8 encoding and operands

32-bit R-type: custom-0 opcode=0x0b, funct3=0, funct7=0x21.
Match=0x4200000b; mask=0xfe00707f.

```asm
.insn r 0x0b,0,0x21,rd,rs1,rs2
```

- rs1[31:0]: signed INT32 accumulator; rs1[63:32] ignored.
- rs2[31:0]: nonnegative INT32 multiplier M (bit31 must be zero).
- rs2[39:32]: signed INT8 shift, valid range [-31,30].
- rs2[47:40]: signed INT8 output zero-point, valid range [-128,127].
- rs2[63:48]: reserved, must be zero.
- rd: INT8 result sign-extended to the full 64-bit register. Old rd is not read.

Parameter packing is explicit software work, not hidden simulator state:

```text
params = uint32(M) | (uint64(uint8(shift)) << 32)
                   | (uint64(uint8(output_zero_point)) << 40)
```

## Exact arithmetic

Snapshot both sources before writeback; decode x=sign32(rs1[31:0]).

```text
b = 31 - shift                                      // b in [1,62]
scaled = floor((x*M + 2^(b-1)) / 2^b)               // mathematical wide result
offset = scaled + output_zero_point
q = min(127, max(-128, offset))
rd = sign_extend_64(INT8(q))
PC = PC + 4
```

This uses the existing single-rounding profile, not double rounding. Ties in
the scaling formula round toward positive infinity (e.g. -0.5 -> 0, +0.5 -> 1).
It is separate from the offline FP32-to-INT8 fixture quantization rule.

All intermediate operations are defined in wide signed arithmetic. For the
specified x/M/shift range, product plus nudge fits INT64; adding the INT8 zero
point to scaled also fits INT64. Do not first truncate scaled to INT32 before
the final clamp. The old software helper's stated domain requires scaled to
fit INT32; results are identical on that domain, including all current workload
and 308 reference cases. requant8 additionally defines wide-scale saturation
outside that software precondition, tested by an arbitrary-precision reference.

Invalid M, shift or reserved parameter bits trap as illegal instruction,
including when rd=x0. Invalid funct3/funct7 combinations do not decode as this
instruction. No memory access, exceptions from memory, CSR state or implicit
quantization configuration is introduced by requant8 itself.

Register overlap rd=rs1, rd=rs2, rs1=rs2 and all equal is legal. x0 reads zero,
and rd=x0 discards valid output. rs2=x0 is a VALID parameter encoding:
M=0, shift=0, zero-point=0, hence output zero. Other XLEN/register profiles use
illegal handlers; only RV64IMC execution is verified here.

## Interface and hardware implications

Unlike qmac8, requant8 has two explicit inputs and does not read old rd. This
fits a two-read-port operand interface conceptually, but RTL implementation,
forwarding, stalls, latency, timing and area are not implemented/measured here.
One retired instruction is NOT evidence of single-cycle target execution.
The parameter packing loads/shifts/ORs are inside the measured kernel interval.

Reference for the baseline profile:
https://github.com/tensorflow/tensorflow/blob/v2.19.0/tensorflow/lite/kernels/internal/common.cc
MultiplyByQuantizedMultiplier, TFLITE_SINGLE_ROUNDING=1 scalar int32 overload.

This instruction performs no GELU, nonlinear approximation, input-zero-point
correction or model calibration. It is a scalar one-output requantization,
despite the 8 in the name referring to its INT8 output.
