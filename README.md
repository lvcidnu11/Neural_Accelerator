# RV64IMC W8A8 DeiT-Tiny Acceleration

Bare-metal quantized inference for the NTHU EE6455 advanced computer architecture
project. This repository implements two experimental RISC-V instructions in
Spike, uses them from C/C++ kernels through LiteRT Micro (TFLM), and runs a
complete 12-block DeiT-Tiny graph with a project integer reference.

**Measured result:** on the full FC1 operator, `qmac8 + requant8` reduces retired
instructions by **88.94%** versus upstream TFLM FC (**9.04× instruction-count
ratio**). In a separate matched-kernel study, register blocking plus FC1/GELU
fusion reduces instructions by another **36.07%** and TFLM arena usage by
**36.69%**. These are functional Spike instruction counts and allocation
measurements; hardware cycle speedup has not been measured.

**Current milestone:** instruction semantics/tests, operator benchmarks and full
synthetic-input inference are verified. CPU integration, technology-mapped
area/timing, dataset accuracy and final slides/report remain open. The repository
is a software/ISA checkpoint, not a claim that all final requirements are done.

## Contents

- [1. Workload and execution flow](#1-workload-and-execution-flow)
- [2. Design choices](#2-design-choices)
- [3. Measured improvements](#3-measured-improvements)
- [4. Reproduce the results](#4-reproduce-the-results)
- [5. Correctness and measurement rules](#5-correctness-and-measurement-rules)
- [6. Repository map](#6-repository-map)
- [7. Limitations and next milestones](#7-limitations-and-next-milestones)

## 1. Workload and execution flow

The model is the official **non-distilled `deit_tiny_patch16_224`**: 12 blocks,
197 tokens, embedding dimension 192, three attention heads, MLP hidden dimension
768, and 1,000 classifier outputs. Matrix weights and activations are INT8;
accumulators, bias and some normalization/residual parameters use wider integers.

```text
Official pretrained checkpoint + deterministic synthetic image (seed 6455)
        │
        ▼  Offline Python: inspect, choose scales, quantize, build reference
FlatBuffer graph + quantized constants + input + expected integer outputs
        │
        ▼  Compile C/C++ kernels + LiteRT Micro + bare-metal startup; link
RV64IMC ELF + Spike custom-instruction plugin
        │
        ▼  Execute and verify; measure a delimited instruction interval
Exact-output checks + retired counts + instruction mix + code/memory sizes
```

The FC benchmark uses upstream TFLM FullyConnected as its scalar baseline and
project FC/GELU registrations for custom variants. Full inference uses TFLM's
interpreter and static arena with **14 project custom nodes**: embedding,
12 complete Transformer blocks and classifier. The graph is generated offline;
it is not a generic builtin-only `.tflite` file for an arbitrary LiteRT runtime.

The shared [Linear kernel](experiments/tflm-linear/linear_kernel.h) is used for
FC1, FC2, QKV, projection, attention matrix products, embedding and classifier.
FC1 is the main incremental benchmark, rather than the only optimized call site.
See [the code walkthrough](docs/code-flow.md) for the path from model to ISA.

## 2. Design choices

| Choice | Why we chose it | Cost or boundary |
|---|---|---|
| RV64IMC, `lp64`, static buffers | Matches the bare-metal target; no floating-point/vector ISA or target OS is needed | Host conversion still uses floating point; memory must fit the fixed layout |
| Signed W8A8 matrix operands; per-channel weight scales | Eight INT8 values fit one RV64 register; per-channel scales retain channel-specific weight ranges | Current activation scales come from a synthetic fixture, not dataset calibration |
| `qmac8`: 8 signed products + old `rd` accumulator | Replaces repeated scalar multiply/add and amortizes loop/control work | Old `rd` is an implicit third input; CPU integration needs operand/forwarding support |
| `requant8`: accumulator + packed parameter register | Performs one-output fixed-point scaling, rounding, zero-point addition and clamp using two explicit sources | Loads and parameter packing remain inside the measured workload |
| INT8 GELU LUT, generated offline | Exactly 256 possible INT8 inputs; runtime becomes an integer lookup | Table depends on input/output quantization; does not recover precision lost before GELU |
| FC1→GELU fusion | Consume the requantized value immediately, avoiding an intermediate traversal and separate GELU dispatch | Specialized to the tested Plan A quantization semantics |
| Compile-time blocking `B=1,2,4` | Reuse one packed input against multiple output channels while keeping accumulators in registers | More registers; tails/alignment need fallback paths; larger B is not assumed better |
| LiteRT Micro integration plus static custom graph | Tests real interpreter/arena/dispatch behavior and reuses the same kernel across model shapes | Full-model LayerNorm/Softmax are project integer implementations |
| Independent integer reference before optimization | Separates arithmetic correctness from performance claims | Matching that reference alone does not establish model accuracy |

The exact encodings, register overlap rules, arithmetic widths and traps are in
[ISA.md](experiments/qmac8-requant-spike/ISA.md). [DESIGN.md](DESIGN.md) explains
alternatives, assumptions and hardware implications.

## 3. Measured improvements

Evidence below comes from [reproduced profiles](results/reproduced/) verified
from this course repository on **2026-10-25**. Raw JSON is retained alongside
[the earlier reference run](results/reference-2026-10-25/). See
[the detailed result tables](docs/results.md) and [measurement protocol](EXPERIMENTS.md).

Reduction is `1 - new / baseline`; instruction-count ratio is `baseline / new`.
Every comparison within a table uses the same data, quantization and execution
scope. Percentages from different baselines must not be added together.

### 3.1 Standalone ISA/kernel checkpoint

Model-derived FC tile: **M=2, K=192, N=8**, 3,072 mathematical MACs.
PATH GCC 16.2.0, O2, same standalone C driver and reference checks.

| Configuration | Retired instructions | Reduction vs scalar |
|---|---:|---:|
| Scalar C | 22,271 | — |
| qmac8 + software requantization | 3,199 | 85.64% |
| qmac8 + requant8 | 2,834 | 87.27% |

This small test verifies real instruction execution. The larger TFLM benchmark
below includes interpreter dispatch and uses a different compiler/driver;
its counts are not directly comparable to the standalone tile.

### 3.2 Full FC1: packed MAC, then custom requantization

**M=197, K=192, N=768**: 151,296 outputs and 29,048,832 mathematical MACs.
Pinned xPack GCC 15.2.0-1, O2; the measured scope is TFLM `Invoke()`.

| Configuration | Retired instructions | Reduction vs upstream | Count ratio vs upstream |
|---|---:|---:|---:|
| Upstream TFLM FC | 239,507,000 | — | 1.00× |
| qmac8 + software requantization | 32,997,457 | 86.22% | 7.26× |
| qmac8 + requant8 | 26,491,730 | **88.94%** | **9.04×** |

Custom requantization gives a further **19.72%** reduction relative to the
qmac8/software-requant configuration. This includes changes to packing,
call overhead and register allocation, not only removal of arithmetic helper
instructions. The result exceeds the assignment's 2× instruction-reduction
threshold for a major FC operator; cycle/area requirements remain separate.

### 3.3 Keep the workload fixed: FC1 → GELU

All rows execute full FC1 **and the same quantized GELU LUT**.

| FC configuration | Invoke retired instructions |
|---|---:|
| Upstream TFLM FC + GELU LUT | 240,566,310 |
| qmac8/software requant + GELU LUT | 34,056,767 |
| qmac8/requant8 + GELU LUT | 27,551,040 |

Adding GELU expands the execution range. These rows do **not** establish the
speedup of LUT GELU versus a non-LUT GELU. The standalone GELU interval costs
1,059,242 instructions in these runs.

### 3.4 Isolate register blocking and fusion

All rows below use **the same shared templated kernel**, qmac8 and requant8,
with full FC1→GELU scope. Its B=1 starting point differs from the older adapter
in §3.3; use this table to isolate blocking/fusion.

| Channels per block | Separate GELU | Fused GELU |
|---|---:|---:|
| B=1 | 26,946,684 | 26,189,963 |
| B=2 | 20,745,728 | 19,913,359 |
| B=4 | 17,871,511 | **17,228,262** |

- **Blocking only:** B=1 separate → B=4 separate reduces instructions **33.68%**.
- **Fusion at B=4:** separate → fused reduces instructions **3.60%**.
- **Combined:** B=1 separate → B=4 fused reduces instructions **36.07%**.
- **Arena usage:** separate 309,584 bytes → fused 196,000 bytes, saving
  **113,584 bytes (36.69%)**. Arena usage is not total target RAM.

Disassembly checks confirm five packed loads and four qmac8 instructions in the
B=4 hot loop, with no stack access inside that loop. Function entry/exit and
outer loops can still access the stack. All 151,296 final outputs match the
same integer reference.

### 3.5 Complete inference: integration result

| Metric | Optimized full DeiT-Tiny |
|---|---:|
| Retired instructions | **862,447,442** |
| qmac8 executions | 156,823,872 |
| requant8 executions | 5,974,636 |
| Mathematical MACs | 1,253,683,200 |
| Checked nodes / complete blocks / final logits | 14 / 12 / 1,000 |
| TFLM arena actually used | 190,416 bytes |

The ROI includes operator dispatch and diagnostic stage-hash checks. This is
one synthetic input. All 1,000 logits match the project integer reference and
14 intermediate 32-bit hashes match. Quantized top-1 is **38**, versus float
top-1 **79**: there is no dataset accuracy claim. A scalar full-model comparison
has not been measured, so no end-to-end speedup is reported.

## 4. Reproduce the results

### 4.1 Supported host and prerequisites

Validated on **macOS arm64**, **Python 3.13**, Apple Clang and the following
tools on PATH:

```sh
python3 --version
clang++ --version
spike --help
riscv64-elf-gcc --version
```

Also required: `git`, `curl`, and `riscv64-elf-{objdump,nm,size,readelf}`. The
validated PATH GCC is 16.2.0. Spike is **1.1.1-dev**, SCM
`55b4658dbf574ba0b714083ec436ce2cb5be1998`; its matching C++ development headers
must be installed. Set `SPIKE_PREFIX` if `include/riscv/extension.h` is outside
Spike's resolved install prefix. The plugin ABI must match the simulator.
Linux/other-host TFLM builds need script adaptation; they are not validated here.

`setup` installs dependencies locally, downloads official model weights and
checks their full SHA256. It pins TFLM commit
`8f6596d99898deaac36ebd8ff8bf1b9754fd2cd1`, xPack GCC **15.2.0-1** with
newlib/C++ headers, GNU Make **4.4.1**, and the Python package lock. No global
Python packages or replacement system compiler are installed. Allow several GB
of disk space and network access for setup. See [reproduction details and
troubleshooting](docs/reproduction.md).

### 4.2 Clone, set up, test and measure

```sh
git clone git@github.com:ScGong04/Neural_Accelerator.git
cd Neural_Accelerator
python3 scripts/project.py setup
python3 scripts/project.py test
python3 scripts/project.py bench
python3 scripts/project.py full
python3 scripts/publish_results.py
```

For an existing checkout, start in its repository root. After setup, use
`python3 scripts/project.py all` to run test → bench → full in order.

| Stage | Output / pass condition |
|---|---|
| `setup` | Local environment, pinned tools/dependencies and verified official checkpoint |
| `fixtures` | Deterministic float capture and independent integer Linear reference |
| `test` | Actual Spike ISA tests and trap controls; host UBSan + Spike tail/blocking/fusion checks |
| `bench` | Scalar/qmac/requant FC profiles, FC1→GELU comparison, B=1/2/4 × fusion ablations |
| `full` | Graph export, host + UBSan checks, RV64IMC build, Spike correctness/profile |
| `publish_results.py` | Small path-sanitized JSON evidence in `results/reproduced/` |

Run stages in order for the full evidence set. `full` can also run after setup:
it generates fixtures and builds its own TFLM libraries/plugin. A stage exits
nonzero on failure and prints its log path. Last validation used local cached
dependencies and reran setup plus all stages; a separate empty-machine clone
has not been exercised.

### 4.3 Build settings and generated artifacts

Target flags: `-march=rv64imc -mabi=lp64 -mcmodel=medany -O2`, no LTO,
`TFLITE_SINGLE_ROUNDING=1`, linker relaxation disabled. Standalone tests use the
PATH GCC; TFLM/full-model builds use pinned xPack GCC. RAM starts at `0x80000000`,
size 32 MiB; stack begins at the exclusive upper boundary `0x82000000` and grows
into RAM. Startup clears BSS, initializes constructors and terminates through
Spike's `tohost` protocol.

```text
build/logs/                         Stage summaries and failure logs
experiments/*/build/                Generated vectors, models, ELF, map, disassembly,
                                   per-run commands, profiles and source hashes
experiments/deit-tiny-inspect/      Local .venv/, weights/, and generated results/
experiments/tflm-linear/vendor/     Downloaded TFLM, Make and xPack compiler
results/reproduced/                 Compact committed measurements, no model binaries
```

Dependencies, weights and generated build outputs are ignored by Git. Existing
legacy binaries under `baseline/smoke` were already tracked and are preserved.
Do not use `git add -f` to include downloads or new build artifacts.

## 5. Correctness and measurement rules

- **Instruction tests:** 1,645 qmac8 vectors plus 33 chain executions; 2,708
  unique requant8 ISA vectors plus 308 paired software/custom checks. Cover
  signed lanes, wrapping, full-register results, overlap/x0, rounding, clamp
  and invalid-parameter/encoding traps, in fast and commit-logged Spike modes.
- **Kernel boundaries:** 20 tail/alignment cases per standalone custom variant;
  640 blocking/fusion cases with independent scalar reference and canaries,
  checked with host UBSan and actual Spike.
- **GELU:** all 256 table entries checked against independent PyTorch float64
  GELU; fused and separate outputs follow the same INT8 boundary semantics.
- **Full inference:** all 1,000 final logits checked exactly; intermediate
  32-bit hashes are diagnostics, not proof of byte-for-byte equality of every
  intermediate tensor.
- **Measurement:** debugger-delimited `minstret` differences and subtracted
  mnemonic histograms must agree. Startup and final comparisons are excluded;
  full-model stage hashes are included. Parameter loads/packing are included.
- **Interpretation:** retired instructions are not cycles. Load/store counts
  are instructions, not transferred bytes or cache misses. No hardware latency,
  CPI, Fmax, physical area or cycle speedup is inferred from Spike.

See [EXPERIMENTS.md](EXPERIMENTS.md) for exact ROI and metric definitions.

## 6. Repository map

| Directory / document | Role |
|---|---|
| `baseline/smoke/` | Original assembly, linking and bare-metal learning checkpoint |
| `experiments/c-dot-smoke/` | Reproducible C startup/linker checkpoint used by standalone tests |
| `experiments/deit-tiny-inspect/` | Official checkpoint validation and model inspection |
| `experiments/deit-linear-int8/` | Scalar C operator and independent integer reference |
| `experiments/qmac8-spike/` | First qmac8 implementation and preserved baseline tests |
| `experiments/qmac8-requant-spike/` | Current combined ISA implementation, wrappers and tests |
| `experiments/qmac-design/` | Standalone one/two/eight-lane RTL exploration |
| `experiments/tflm-linear/` | TFLM FC/GELU integration, shared kernel and optimization study |
| `experiments/deit-full-int8/` | Full model export, custom graph operators and execution |
| `scripts/` | Repository-level reproduction and evidence export |
| `results/` | Compact reference/reproduced evidence and provenance |
| [DESIGN.md](DESIGN.md) | Design rationale, numerical contract and alternatives |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Workload/configuration/measurement protocol |
| [docs/results.md](docs/results.md) | Additional metric tables and baseline-specific reductions |
| [docs/requirements.md](docs/requirements.md) | Assignment mapping and remaining deliverables |
| [AI_USAGE.md](AI_USAGE.md) | Assistance disclosure and attribution |

Start reading [linear_kernel.h](experiments/tflm-linear/linear_kernel.h), then
[fc_adapter.cc](experiments/tflm-linear/fc_adapter.cc) and
[model.cc](experiments/deit-full-int8/model.cc). The actual instruction handlers
are in [extension.cc](experiments/qmac8-requant-spike/extension.cc).
Experiment directories preserve staged implementations; full inference reuses
the shared kernel rather than the historical standalone kernels.

## 7. Limitations and next milestones

1. Establish the assignment's software-optimized **scalar B1** and a structured
   comparative second-operator study. Template blocking B=1 is not scalar B1.
2. Add full-model operator attribution and a fair scalar full-model baseline.
3. Calibrate/evaluate on representative images; current synthetic scales and
   differing float/quantized top-1 cannot support accuracy claims.
4. Integrate the instructions into CV-Wally, verify operand hazards/retirement,
   measure cycles/CPI, and synthesize using the required technology library.
   Standalone RTL generic cell counts are not SAED physical area or CPU speedup.
5. Complete slides and the final report using those measurements.

The [assignment](rv64imc_quantized_nn_term_project.md),
[requirement checklist](docs/requirements.md),
[commit organization and push instructions](docs/submission.md), and
[relocation verification](docs/relocation-verification.md) record the scope.
