# Design

## Dataflow and runtime

Official non-distilled DeiT-Tiny has 12 blocks, 197 tokens, embedding dimension
192, three attention heads and MLP hidden dimension 768. Offline Python loads
pretrained weights, captures a deterministic seed-6455 synthetic input, chooses
project scales, quantizes matrix weights per output channel, and generates an
integer reference plus a FlatBuffer graph. The C++ program statically links
LiteRT Micro (TFLM), the graph, constants and input into a bare-metal ELF.

The FC experiment uses the upstream TFLM FullyConnected registration for the
baseline and project adapters for custom variants. Full inference uses TFLM's
interpreter/arena with 14 project custom nodes: embedding, 12 complete blocks,
and classifier. Each block executes LayerNorm, QKV, attention scores, integer
Softmax, attention-value multiplication, projection, residual, LayerNorm, FC1,
GELU and FC2/residual. It is a project static graph and custom runtime; it is
not a generic builtin-only model runnable by any LiteRT interpreter.

W8A8 applies to matrix operands. Bias/accumulators and normalization/residual
parameters use wider integers. LayerNorm uses integer square root; Softmax uses
an offline exponential table and integer normalization. Their rules are project
semantics, not a claim of bit-exact equivalence to arbitrary TFLite operators.
Scales come from synthetic activations, not representative calibration data.

## Packed MAC

`qmac8 rd, rs1, rs2` takes two RV64 registers containing eight signed INT8 lanes.
It adds their eight products to the old low 32 bits of `rd`, wraps modulo 2^32,
and sign-extends the result. Thus old `rd` is an implicit accumulator input.
Both registers carry values rather than addresses. Loads and pointer updates
remain separate instructions. This reduces instruction work without implying
that the simulator's functional handler is a single-cycle hardware unit.

## Requantization

`requant8 rd, rs1, rs2` uses the signed low 32-bit accumulator from `rs1` and packed
parameters from `rs2`: multiplier bits 31:0, signed shift bits 39:32, signed output
zero point bits 47:40; upper bits must be zero. Multiplier is nonnegative;
shift is -31 through +30. For `bits = 31 - shift`:

```
y = floor((acc * multiplier + 2^(bits-1)) / 2^bits)
q = clamp(y + output_zero_point, -128, 127)
```

The arithmetic is wide before clamping. This defines negative ties as well as
positive ones; offline quantization of weights and reference GELU tables has
its own explicitly implemented rounding. Invalid parameter encodings trap.
See the authoritative [ISA contract](experiments/qmac8-requant-spike/ISA.md).

## GELU, fusion and register blocking

Plan A requantizes FC1 INT32 outputs to INT8, then indexes a 256-entry table.
Offline `LUT[q] = quantize(GELU(input_scale*(q-input_zero_point)))` incorporates
input/output scales. Runtime is integer lookup; table generation is checked
against PyTorch float64 for all 256 inputs. This preserves Plan A semantics,
including the precision loss from the intermediate INT8 quantization.

Fusion computes the same requantized INT8 value, immediately looks it up and
writes the final output. It removes the intermediate tensor traversal and
separate operator dispatch without changing the integer reference result.

`template<int B> linear` specializes B=1, 2 or 4 output channels per token.
The same packed input register is reused against B different weight rows
(logical columns in mathematical W[K,N]). Four accumulators stay in registers
for B=4. The checked hot loop has five packed loads and four qmac8 instructions,
with no stack access inside that loop; outer loops/prologues may use the stack.
Scalar paths handle unaligned data and dimension tails. `noinline` keeps a
visible function boundary for inspection; B is a compile-time parameter.

## Hardware boundary

Standalone one/two/eight-lane qmac RTL candidates exist under
`experiments/qmac-design`; they are not integrated into a CPU. Generic synthesis
cell counts do not satisfy SAED physical area/Fmax requirements. Required future
work includes decode/hazards, old-rd operand handling, precise retirement,
requantization RTL, core verification and fair cycle/area experiments.

## Decision record and alternatives

### Why begin with operators, then integrate the model?

A small model-derived Linear tile makes accumulator, signedness and rounding
errors visible without the rest of the network. The full FC1 benchmark then
adds representative dimensions and TFLM overhead. Only after these checks does
the full graph exercise different Linear shapes, attention tails and shared
buffers. This makes failures localizable; the tile alone is insufficient
performance evidence, and full-model agreement alone would hide which
instruction or operator failed.

### Why a register-register qmac rather than a memory instruction?

Explicit loads keep address calculation, alignment, memory faults and memory
ordering in existing instructions. qmac8 only transforms register values, with
no hidden pointer or scale state. An alternative fused load/MAC instruction
might save instructions, but would add memory exceptions and integration
complexity. No such alternative is measured here.

RV64 naturally holds eight INT8 lanes. The arithmetic contract is modulo 2^32,
not saturation on each MAC. Full-model conversion checks conservative bounds
`abs(bias) + 128*sum(abs(weight)) < 2^31` per output channel and verifies actual
reference accumulators, so the tested model does not rely on wraparound for
its numerical result. ISA tests separately verify wrapping semantics.

Old `rd` provides accumulation while preserving two explicit sources in the
encoding. It does not remove the hardware need to obtain three operand values.
A CPU implementation must address register-file reads, bypassing and hazards;
using the functional Spike implementation does not solve those problems.

### Why pack requantization parameters instead of using several instructions?

Each channel needs an accumulator, multiplier, shift and output zero point.
The latter three fit in one RV64 register, leaving a two-explicit-source
instruction. This avoids a persistent configuration CSR and makes per-channel
parameters explicit. Parameter loads, shifts and ORs are part of the ROI;
performance is not measured with free simulator-side parameters.

A configuration instruction plus stateful requant operation is an alternative,
but would require defining state lifetime, dependencies and context handling.
It has not been implemented. The current instruction returns one INT8 result;
`requant8` does not mean eight accumulators are requantized in parallel.

### Why this rounding profile?

We fix a single-rounding Q31 contract and match the software reference on its
valid domain. The wide-scale saturation behavior is also explicitly defined
outside the old helper's INT32-scaled-result precondition. A double-rounding
scheme, different tie rule or saturating intermediate shift could produce
different INT8 outputs. These are numerical contracts, not interchangeable
performance options. The tests include negative ties and extreme shifts.

### Why Plan A LUT rather than direct INT32-accumulator GELU?

Plan A establishes a finite INT8 interface: requantize first, then apply the
scale-specific 256-entry table. That permits exhaustive table checking and
makes separate/fused operators exactly comparable. A direct accumulator-aware
GELU could avoid the first quantization's precision loss, but its input domain
and approximation are different. It would need an independent accuracy study;
it is not what the current fusion implements.

### Why B=4, and what does the weight layout mean?

Mathematically `Y[M,N] = X[M,K] * W[K,N]`. Storage is output-channel-major:
`w[c*K+i]` holds mathematical `W[i,c]`. Thus computing four output channels
reads four distinct contiguous weight rows in storage while sharing one input
row. Different columns of mathematical W are not identical; W is reused across
tokens, not across output channels.

For four eight-element dot-product chunks, B=1 needs eight packed operand loads
(four input plus four weight); B=4 needs five (one input plus four weight).
All four qmac8 operations still execute. This is a local load-count argument,
not a claim of 37.5% less total model memory traffic. Scalar tails, parameters,
output stores, dispatch and other operators remain.

B=1/2/4 are measured rather than extrapolated. More blocking could exceed the
available registers or worsen code size, spilling or scheduling. The measured
B=4 hot loop has no stack accesses, which supports its intended reuse on this
compiler; it is not a guarantee for another compiler or larger B.

### Why TFLM custom nodes rather than a generic full-model converter?

TFLM supplies FlatBuffer handling, operator registration, static allocation and
Invoke dispatch. Custom nodes let the project define integer attention,
normalization and fused Linear behavior without adding a target OS or generic
floating-point runtime. The tradeoff is a specialized graph/runtime pair and
project numerical semantics. The 14 nodes are not a count of primitive
mathematical operations: each block node executes an entire Transformer block.
This design demonstrates integration but does not claim generic LiteRT builtin
compatibility or calibrated ImageNet accuracy.

## Standalone RTL design space

The preserved RTL experiment checks 1,042 vectors for each design. Its measured
unit latency and generic synthesis cell counts are:

| Parallel INT8 multipliers | Unit latency (cycles) | Generic combinational cells | Generic sequential cells | Total generic cells |
|---|---:|---:|---:|---:|
| 1 | 8 | 851 | 197 | 1,048 |
| 2 | 4 | 1,446 | 197 | 1,643 |
| 8 | 1 | 4,742 | 197 | 4,939 |

Source: [RTL summary](experiments/qmac-design/results/summary.json).
More lanes trade generic logic cost for fewer standalone arithmetic cycles.
Clock period, critical path, technology-mapped area, issue/retirement overhead
and CPU stalls are unmeasured, so these numbers cannot select the fastest CPU
implementation or establish speedup/area. Requantization RTL is still needed.
