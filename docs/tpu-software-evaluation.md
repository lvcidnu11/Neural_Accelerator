# TPU and software evaluation, 2026-11-02

## Decision

Keep the TPU branch as a promising matrix-offload exploration. It reduces core
instruction work and logical read bytes on the tested model, but the project
has not established hardware speedup or a favorable performance/area ratio.
The original result compares against the existing qmac8/requant8 path with
B=4 blocking and FC1/GELU fusion, rather than against scalar RV64IMC.

Keep two optional software changes: inline Linear, and hoist bias/requant
parameters across rows in each TPU tile. The latter gives the larger reduction.
Both preserve the existing arithmetic, model, input and quantization.

## Controlled full-model ablation

Same GCC 15.2.0, O2, RV64IMC/lp64, pinned TFLM library, model and synthetic
input. Each ROI includes all 14 diagnostic stage hashes and dispatch.
Every variant passes the host UBSan model and Spike model checks: 14 stage
hashes and all 1000 final logits. No dataset accuracy claim follows.

| Configuration | Core retired instructions | ELF .text bytes | Reduction vs own original path |
|---|---:|---:|---:|
| qmac8/requant8, B=4, fused GELU | 862,447,442 | 76,672 | reference |
| qmac path + always_inline Linear | 830,264,926 | 76,824 | 3.7315% |
| Original TPU path | 262,234,932 | 76,104 | reference |
| TPU + always_inline Linear | 256,453,645 | 76,520 | 2.2046% |
| TPU + parameter hoisting | 220,234,394 | 76,148 | 16.0164% |
| TPU + parameter hoisting + inline | 214,329,448 | 76,540 | 18.2682% |

The final TPU path reduces core instructions by 74.1854% versus the newly
inlined qmac path, or 75.1487% versus the original qmac path. These comparisons
have different denominators and should be labeled accordingly. Do not multiply
FC1-only ratios by full-model ratios.

## What inline actually changes

The qmac8/requant8 instruction wrappers already use always_inline. The shared
Linear function previously used noinline. This experiment removes that boundary
with always_inline, allowing call-site specialization as well as call removal.
The savings therefore do not measure only the call/return instructions.

qmac8 remains 156,823,872 and requant8 remains 5,974,636 in the inlined qmac
run. The mathematical products and requantizations are unchanged. Remaining
reductions come from control, address computation, loads and other generated
integer work. A disassembly-level study would be needed to isolate each
compiler transformation. Normal C++ inline is not a guarantee of expansion.
See GCC: https://gcc.gnu.org/onlinedocs/gcc/Inline.html.

## Why parameter hoisting helps

Original TPU postprocessing visits rows first, then channels. It reloads bias,
multiplier and shift and constructs requant8 parameters for each output.
The new loop visits channels first, builds parameters once per channel in a
tile, then processes up to eight rows. It keeps the exact INT8 boundary before
GELU. This changes CPU software only. All TPU variants still execute 95,225
commands, 1,255,044,864 padded products, 322,601,216 engine read bytes and
23,898,544 engine write bytes. The proposed engine hardware is unchanged.

The final TPU path has 42,965,295 CPU load instructions, 26,088,693 stores,
26,538,849 conditional branches and 5,974,636 requant8 instructions.
CPU logical reads/writes are 94,376,217 / 35,022,048 bytes. Including the
engine gives 416,977,433 logical read bytes and 58,920,592 logical write bytes.
For the inlined qmac comparison these are 1,684,621,609 read and 26,722,248
write bytes. Logical reads fall 75.25%, but writes rise 120.49%.
These are sums of executed operand widths, not DRAM traffic or cache misses.

## Correctness and evidence

The retained qmac inline path also passes 640 independent dimension, tail,
alignment, blocking and fusion cases on host UBSan and Spike. The retained
TPU path passes 900 dimension/alignment/fusion cases plus an extrema case.
These supplement, rather than replace, the original ISA negative controls.
Input seed is 6455. The kernels compare against an independent scalar oracle.
Raw exact commands, debugger minstret snapshots, PC histogram differences,
ELFs and disassembly live in the evaluation build directory. Selected profiles
assert histogram totals equal minstret differences. Source, ELF and input/model
hashes accompany the compact results in results/software-2026-11-02/.

## Other optimization opportunities

1. Integrate qmac8/requant8 into CV-Wally and measure cycles, CPI, critical path
   and SAED area. This is required course work and the largest evidence gap.
2. Calibrate activation scales on representative data and evaluate accuracy.
   Current integer top1=38 differs from FP32 top1=79 on the synthetic fixture.
3. Add operator boundaries for LayerNorm, softmax, tensor transforms, residuals
   and diagnostic hashes. Report a second major operator explicitly and rank instruction contributions before selecting
   another software or hardware extension. The residual integer bucket cannot
   establish a named operator bottleneck.
4. Explore a fused tile epilogue only as a new architectural hypothesis. It
   could avoid INT32 tile stores/loads and repeated CPU requantization, but it
   adds per-channel scaling/LUT access, interface complexity and hardware cost.
5. Explore larger tiles, persistent weight buffers, double buffering or DMA
   overlap only with buffer capacity and memory timing modeled. None has been
   implemented or credited in the reported result.

Do not add more ISA features solely to improve retired-instruction counts.
The next major project milestone is evidence about correctness/accuracy and
hardware cost, rather than a larger instruction-count ratio.

## Reproduction

Existing bootstrap, libraries and full-model fixtures are prerequisites.

```sh
# Controlled six-way ablation, with a frozen a2bee01 Linear source
python3 experiments/tpu-evaluation-2026-11-02/evaluate.py --repo .
python3 experiments/tpu-evaluation-2026-11-02/evaluate.py --repo . --variants qmac-inline tpu-inline-hoist --histograms
python3 experiments/tpu-evaluation-2026-11-02/kernel_checks.py qmac-inline tpu-inline-hoist
python3 experiments/tpu-evaluation-2026-11-02/export_results.py

# Integrated optional build, original defaults remain available
python3 experiments/deit-full-int8/build.py --target riscv --inline-linear
python3 experiments/deit-full-int8/profile_spike.py --inline-linear
python3 experiments/deit-full-int8/build.py --target riscv --tpu --inline-linear --hoist-tpu
python3 experiments/deit-full-int8/profile_spike.py --tpu --inline-linear --hoist-tpu
```

AI assistance: Codex assisted with the ablation, kernel changes, measurement
scripts, analysis and presentation. The student should review the arithmetic,
measurement boundaries and hardware limitations before presenting.

## Memory-cost follow-up

The [inline memory audit](inline-memory-evaluation.md) isolates inline from
parameter hoisting. qmac inline adds 152 code bytes with no observed stack
increase. After TPU hoisting, inline adds 392 code bytes and 200 observed
stack bytes for 2.68% fewer instructions. Static data and arena use stay
unchanged. A functional I-cache sensitivity study records a small miss
increase on TPU. Keep inlining optional for targets with a tight code/stack
budget. The stack canary measures this fixture's written depth, not a formal
worst-case bound.
