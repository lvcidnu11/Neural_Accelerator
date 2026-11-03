# Complete-inference comparison for the five-minute presentation

## Scope and controls

All three configurations execute the same complete non-distilled DeiT-Tiny
integer model: embedding, 12 Transformer blocks and the 1000-logit classifier.
The learned matrices, model FlatBuffer, input (seed 6455), quantization and
reference outputs are identical. Each ROI includes TFLM Invoke and fourteen
stage-hash checks, and excludes preparation and final-logit verification.
GCC 15.2.0, O2, RV64IMC/lp64 and the TFLM library remain fixed.

The scalar baseline uses direct INT8 scalar multiply/add and software fixed-point
requantization. It retains B=4 input reuse, the same GELU LUT/fusion and Linear
inlining. It does not inefficiently emulate packed qmac8 by shifting registers.
This is an optimized scalar ISA control, corresponding more closely to B1
than an unoptimized B0. The full-model scalar control is newly measured here.
Earlier 862M and 830M configurations already use qmac8/requant8 and must not
be labeled as a scalar baseline.

## Complete-model results

| Configuration | Core retired instructions | Ratio versus scalar | .text bytes |
|---|---:|---:|---:|
| Scalar RV64IMC, B=4/fused LUT/inline | 6,095,818,622 | 1.0000 | 77,424 |
| Our complete CPU design, including inline | 830,264,926 | 7.3420 | 76,824 |
| TPU tile extension, parameter reuse + inline | 214,329,448 | 28.4413 | 76,540 |

Our CPU design reduces retired instructions by 86.3798% versus the optimized
scalar control. The TPU version reduces them by another 74.1854% versus our
CPU design, or 96.4840% versus scalar. These are CPU instruction-count ratios,
not measured hardware speedups. A tile command hides substantial internal
matrix arithmetic and memory work from CPU retirement.

## Everything included in our design

- qmac8 packed signed INT8 dot-product instruction.
- requant8 fixed-point scaling, rounding, zero point and saturation instruction.
- C++ template register blocking, B=4 in the measured final configuration.
- Aligned packed operand loads and correctness-preserving tail/alignment fallback.
- A 256-entry quantized INT8 GELU lookup table.
- FC-to-GELU fusion, preserving requantization before the LUT.
- Selective Linear inlining, in addition to existing inline instruction wrappers.

The shared complete-model implementation additionally supplies integer LayerNorm,
Softmax LUT, residual rescaling and static memory allocation through a 14-node
custom TFLM graph. Those common workload functions are not individually credited
with measured performance gains in this comparison.

## The two meanings of TPU reuse

Matrix operand reuse is inside the engine: weight rows and activation rows are
staged once per output tile, then reused across at most eight input rows and
eight output channels. This replaces the CPU dot-product loop, not the whole
inference pipeline. The proposed clocked 8x8 array has not been implemented in RTL.

Parameter reuse is CPU software: bias/multiplier/shift depend on output channel.
The epilogue prepares them once per channel and uses them for up to eight rows.
It retains the original bias, requant8 and optional GELU arithmetic. This adds
no new TPU arithmetic hardware and is separate from matrix operand reuse.

## Validation and reproducibility

All three full-model configurations match fourteen Python diagnostic hashes
and all 1000 final logits. The new scalar path also passes 640 independent
blocking/tail/alignment/fusion cases on host UBSan and stock RV64IMC Spike.
The full scalar model passes host UBSan and stock Spike. Its measured histogram
contains no qmac8, requant8, tile, F/D/V or atomic instructions. Histogram totals
equal debugger minstret differences. Commands, hashes, disassembly and raw logs
are retained in this experiment's build directory.

Integrated reproduction after the existing setup and model export:

```sh
python3 experiments/deit-full-int8/build.py --target riscv --scalar-baseline --inline-linear
python3 experiments/deit-full-int8/profile_spike.py --scalar-baseline --inline-linear
python3 experiments/deit-full-int8/build.py --target riscv --inline-linear
python3 experiments/deit-full-int8/profile_spike.py --inline-linear
python3 experiments/deit-full-int8/build.py --target riscv --tpu --inline-linear --hoist-tpu
python3 experiments/deit-full-int8/profile_spike.py --tpu --inline-linear --hoist-tpu
```

The five-minute deck contains six slides and about 580 spoken English words,
with timing guidance and English notes. It removes FC1-only performance ratios
and the mixed-scope summary page. It lists the complete CPU design rather than
using qmac8/requant8 as shorthand for every optimization.

This remains one synthetic fixture with project quantization. Dataset accuracy,
representative calibration, CV-Wally cycles/CPI, Fmax and SAED area remain open.
AI assistance is recorded in AI_USAGE.md.

## CV-Wally compatibility status

The current custom paths cannot execute directly on the inspected stock
Wally checkout. Its unchanged controller RTL rejects qmac8, requant8
and xtpu, while accepting the standard ADDI/MUL/LD controls. This is a
decoder component test only. No tile RTL, memory adapter or integrated
Wally custom-kernel execution exists yet. The 16 B/cycle exploratory
transfer assumption also is not established for the inspected 64-bit AHB
path. See [the compatibility audit](cvw-compatibility-audit.md) for source
hashes, exact commands, memory/fault concerns and required next evidence.
TPU remains a Spike-only architectural exploration.
