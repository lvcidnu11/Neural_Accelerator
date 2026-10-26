> Repository entry point and current status: [root README](../../README.md).
> Experiment-specific results below retain their stated measurement scope.

# TFLite Micro full fc1 on bare-metal RV64IMC

## 2026-10-25: fusion and register blocking implemented

The same fc1 -> requantization -> GELU mapping is benchmarked with blocking
factors 1/2/4 and separate/fused execution. See build/optimizations.md,
optimizations.json and optimizations-manifest.json. The matched block-1 control
uses the same project::linear template as blocks 2/4; the older adapter remains
available for historical comparison.

| Blocking factor | Separate operators, Invoke retired | Fused, Invoke retired |
|---|---:|---:|
| 1 | 26,946,684 | 26,189,963 |
| 2 | 20,745,728 | 19,913,359 |
| 4 | 17,871,511 | 17,228,262 |

Combined reduction versus the matched control: 36.0654%. Versus the earlier
27,551,040 adapter, reduction is 37.4678%, which also includes refactor effects.
Fusion is isolated by comparing the two columns within the same blocking factor.
Every configuration produces the same 151,296 final INT8 outputs. Fused runs
do not allocate/store the fc1 intermediate; unfused runs additionally check it.
The table assumes the current equal fc1/GELU output scales and zero points;
the fused registration is specialized to this exported quantization domain.

Fused tensor arena usage is 196,000 bytes versus 309,584 bytes unfused. The
block-4 hot loop has five packed loads, four qmac8 instructions and no stack
access, verified from disassembly (build/blocking4-hot-loop.txt). Outer-loop
and function-entry/exit stack traffic still exists. This does not measure caches
or hardware cycles.

Code: linear_kernel.h is the shared blocked/fused arithmetic kernel.
fc_adapter.cc integrates it into TFLM; the fused graph uses FC_GELU_LUT custom
registration and preserves requantization before lookup. The all-256-input LUT
checks remain outside the ROI. Host UBSan and actual Spike pass 640 independent
dimension/tail/alignment/fusion tests. The two full matrix shapes pass native
and Spike output checks for all blocking/fusion combinations.

```sh
python3 benchmark_optimizations.py
python3 report_optimizations.py
```

Full 12-block inference now lives in ../deit-full-int8. Its quantization and
accuracy limitations are documented there; the older 'next steps' below are
historical checkpoints.

Verified 2026-10-24. One complete DeiT-Tiny fc1 operator executes through the
real TFLite Micro (TFLM) interpreter. This is not full DeiT conversion, a
complete MLP/Transformer block, calibration or accuracy evaluation.

Latest extension: the optional `--gelu` flow now executes full fc1 -> GELU LUT
as a two-operator TFLM graph. A qmac-only variant isolates custom requantization
benefits. The original fc1-only results below remain valid; see the following
section for the new graph and ablation.

## GELU LUT and incremental comparison

```sh
python3 run.py
python3 run.py --gelu
python3 compare_stages.py
```

Both runs include baseline, qmac-only and qmac8+requant8 on the tile and full
fc1. The second run adds the SAME GELU LUT to all three variants, providing
a fair fixed-workload comparison. Evidence is in build/comparison-gelu.json,
manifest-gelu.json, run-gelu.log (when redirected by the caller), and
build/stages.md / stages.json. GELU builds use the `*-gelu/` directories.

| Requested build-up | Scope | Invoke retired |
|---|---|---:|
| qmac8 + software requantization | Full fc1 | 32,997,457 |
| qmac8 + requant8 | Full fc1 | 26,491,730 |
| qmac8 + requant8 + GELU LUT | Full fc1 -> GELU | 27,551,040 |

Custom requantization reduces full fc1 instructions by 19.7158% relative to
qmac-only. The third row performs additional computation: GELU adds 1,059,310
Invoke instructions, of which 1,059,242 lie in the GELU interval. The extra
68 instructions are the changed dispatch/call interval outside operator markers.

For the identical full fc1 -> GELU workload, counts are 240,566,310 (upstream
FC + LUT), 34,056,767 (qmac8/software requant + LUT), and 27,551,040
(qmac8/requant8 + LUT). GELU costs 1,059,242 instructions in every variant.
There is no measured on-target direct-formula GELU baseline, so these numbers
do not establish a standalone GELU speedup.

gelu_lut.cc registers the project `GELU_LUT` custom graph operator because
the pinned TFLM version has no builtin GELU registration. This is a runtime
operator implemented using ordinary integer instructions, not a new ISA
instruction. Model custom options carry the 256-byte INT8 table plus a
20-byte versioned header (`GLU1`, little-endian input/output FLOAT32 scales and
INT32 zero points). Init validates the payload length/magic; Prepare checks
tensor shapes, INT8 types and agreement between tensor quantization and the
table's declared quantization. The opaque payload is project-specific and
requires this registration; it is not a generic LiteRT GELU builtin model.

Offline generation uses exact-form GELU, `0.5*x*erfc(-x/sqrt(2))`, matching
the source model's approximate=none choice. This is Plan A: fc1 requantizes
INT32 to INT8 first; GELU dequantization/quantization is encoded into the table.
Input/output scales are both 0.0976090282201767 and zero points are zero for
this fixture. GELU output uses nearest/ties-away rounding then INT8 clamp;
the preceding requantization retains its separately specified single-rounding
rule. These are synthetic operator settings, not representative calibration.

Verification uses independent PyTorch FLOAT64 exact GELU and quantization:
all 256 possible inputs match the erfc-generated table. The actual C lookup
function is exercised on all 256 entries outside the measured ROI. Host TFLM,
stock-Spike baseline, qmac-only and combined custom versions check all 151,296
fc1 intermediate outputs and all 151,296 final GELU outputs. The two-operator
graph retains the fc1 buffer through the final operator; a diagnostic pointer
permits post-Invoke checks without changing TFLM's memory planner. That pointer
store and an operator-call counter are included in the measured GELU region.

The compiled LUT loop uses 7 instructions per element; the full GELU interval
is `7*151296 + 170`. LUT payload is 256 bytes; graph/node/quantization metadata
add further bytes (model 169,728 -> 170,192 bytes). Arena use becomes 309,584
bytes due to intermediate/output tensor lifetimes, not just the table size.
Full combined test ELF .text is 87,176 bytes; .rodata includes expected fc1
and GELU outputs, and is not a deployment-only memory measurement.

The next missing MLP stage is full fc2. GELU LUT does not replace fc1
requantization and does not by itself complete the MLP or Transformer block.

## Flow and scope

```
Pretrained-weight/synthetic-activation fixture
  -> export_model.py -> one-operator model.tflite
  -> model_data.h embeds the FlatBuffer bytes
  -> TFLM MicroInterpreter -> FullyConnected kernel
  -> RV64IMC/lp64 executable -> Spike
```

Full fc1: X[197,192], W[768,192], bias[768], Y[197,768]. It computes
29,048,832 MACs and checks all 151,296 INT8 outputs. The old 2x8-output tile
is retained as a smoke test. Weights are actual pretrained weights; activations
and scales come from one synthetic model input, not representative images.

The exporter constructs a TFL3 FlatBuffer with INT8 FullyConnected v4,
per-channel weight scales, INT32 bias, zero zero-points, and no fused activation.
It does not perform PyTorch-to-LiteRT full-graph conversion. Model bytes are
embedded as data; the interpreter and kernels provide the executable code.

## Reproduce

Prerequisite: the parent deit-tiny-inspect project venv and saved float fixture.
First-time dependencies (network needed):

```sh
cd experiments/tflm-linear
python3 bootstrap.py
```

bootstrap.py currently supports macOS arm64. It downloads into vendor/:
TFLM revision 8f6596d99898deaac36ebd8ff8bf1b9754fd2cd1 and its verified
dependencies, GNU Make 4.4.1, and xPack RISC-V GCC 15.2.0-1. It installs
flatbuffers==25.12.19 and tflite==2.18.0 into the existing project venv.
No global toolchain is changed. Download SHA256 values are pinned in the script;
TFLM's downloader checks its own dependency archive checksums and patches.

The existing Homebrew GCC lacks C++ headers/newlib. Both new benchmark variants
use the same local GCC 15.2.0, O2, RV64IMC/lp64, medany, no LTO, no exceptions,
no RTTI, and TFLITE_SINGLE_ROUNDING=1. Startup and linker are shared, with
32 MiB Spike RAM, BSS initialization and C++ init-array calls.

Then rebuild and run offline:

```sh
python3 run.py
```

This exports both shapes, executes upstream TFLM on the host, builds both RV64
variants, checks every output, profiles with the same extension-loaded Spike,
and additionally verifies each baseline on stock Spike. The plugin is rebuilt
from the previously ISA-tested qmac8-requant-spike/extension.cc source.
MODEL_PYTHON can select the project Python environment.

## Measured results

| Full fc1 metric | Upstream TFLM FC | Custom FC |
|---|---:|---:|
| Invoke retired instructions | 239,507,000 | 26,491,730 |
| Operator retired instructions | 239,506,838 | 26,491,554 |
| Invoke instructions/output | 1,583.036 | 175.099 |
| Dynamic loads | 59,158,495 | 8,021,900 |
| Dynamic stores | 151,780 | 151,338 |
| Conditional branches | 30,259,499 | 4,239,053 |
| qmac8 count | 0 | 3,631,104 |
| requant8 count | 0 | 151,296 |
| Full ELF .text bytes | 84,944 | 85,532 |
| TFLM arena used bytes | 196,000 | 196,000 |
| Matching INT8 outputs | 151,296/151,296 | 151,296/151,296 |

Instruction-count ratio: 9.0408x; reduction: 88.9391%. The small-tile Invoke
count is 26,080 -> 3,425. These are new measurements, not extrapolations.
Do not compare directly with the older standalone GCC 16 tile as a measurement
of framework overhead: compiler, baseline code and instrumentation also differ.
This baseline is upstream TFLM reference FC; a systematic B1 software-optimized
baseline remains pending. No best-software claim is made.

No cycles, CPI, physical-unit utilization, area, Fmax or hardware speedup is
measured. Fewer load instructions do not imply proportionally fewer bytes.
qmac8 still requires core register-file/pipeline integration in hardware.

## Code tour

- main.cc: load model, resolver/interpreter, AllocateTensors, copy input,
  Invoke, check all outputs and exact prepared multipliers/shifts.
- fc_adapter.cc: preserves upstream Init/Prepare. Baseline calls upstream Eval;
  USE_CUSTOM_FC uses qmac8/requant8 for the supported INT8 case. Other types,
  zero points, fused activation limits and parameter ranges use upstream
  fallback. The adapter is verified for these exported fixtures; broader model
  integration needs additional domain and correctness checks.
- export_model.py: exact FLOAT32-scale integer reference, FlatBuffer export,
  hashes, and INT32 overflow proof. Full fc1's conservative bound for every
  possible INT8 input is 1,077,580 < INT32_MAX. Scales serialized to FLOAT32
  change zero reference outputs compared with the earlier fixture.
- build.py: source selection for the minimal upstream library, recorded flags
  and object cache. Vendor checkout is unmodified. Delete build/host and
  build/riscv if upstream/dependency headers change, to invalidate the cache.
- platform.cc: diagnostics/newlib support. No heap is supplied. A static
  512 KiB tensor arena is reserved, of which 196,000 bytes are used.
- profile.py: scalable profiling without multi-GB commit logs.

The graph still contains FullyConnected. Custom CPU instructions are internal
to its implementation; no custom graph operator is required.

Preparation uses software FP helpers to compute quantization parameters on
RV64IMC. The measured Invoke region executes integer instructions; its PC
histogram contains no F/D/V/atomic instruction. Generic float paths/software FP
helpers can remain linked in the binary. This is not an integer-only whole
binary claim, and does not require floating-point hardware.

Expected outputs are embedded for verification. Therefore .rodata and total
ELF memory include test data, not just a deployment footprint. Unlike the older
standalone experiment, neither new variant stores an INT32 verification matrix;
both comparisons use identical INT8 output checking outside the measured ROI.

## Boundaries and evidence

Invoke ROI: invoke_begin inclusive to invoke_end exclusive. Includes the
Invoke call, dispatch, computation and returns. Excludes AllocateTensors,
preparation, input copy, output checks, startup and tohost. The nested
operator_begin/end interval includes the Eval wrapper and kernel.

profile.py reads minstret using the host debugger at exact labels; it adds no
target CSR instructions. Separate deterministic runs collect PC histograms
before/after Invoke; subtraction gives the ROI mix. Totals must equal the CSR
delta, and entry/exit marker execution is checked. The installed Spike's
--instructions=N executes N+1 instructions; the script compensates and checks
exact totals every time. Changed Spike behavior fails assertions. This method
was cross-checked against the earlier 22,271-instruction commit-trace tile.

A separate correctness run checks main returns zero, failure_stage is zero,
outputs_checked is complete, and custom_calls is 0/1 as expected, then reaches
tohost. Host TFLM independently matches the Python integer reference.

Evidence: build/comparison.json, manifest.json, run.log; each shape/variant has
profile.json, build.json, ELF, disassembly, map, size, debugger/histogram logs
and exact commands. The manifest records source/data/plugin hashes, upstream
revision, dependency stamps and Python versions. build/ and vendor/ are ignored.

Next: GELU LUT and full fc2 to complete the MLP, then remaining operators to
complete a Transformer block. A full fc1 result does not satisfy those scopes.

AI assistance: Codex helped author the integration/export/profiling scripts
and run verification; the student should review and be able to explain them.
