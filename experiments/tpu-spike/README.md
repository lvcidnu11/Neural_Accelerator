# TPU-inspired 8×8 tile engine

This branch adds `xtpu`, a synchronous INT8 matrix-dot command in functional
Spike, and routes the shared Linear kernel through it when built with
`USE_TPU`. The existing qmac8/requant8 path remains the default. The experiment
uses the same full DeiT-Tiny model, input, scales, bias, GELU tables and stage
checks as the baseline. It does not use a Google TPU service or device.

The design borrows the matrix array and local operand reuse idea from
[Jouppi et al., ISCA 2017](https://research.google/pubs/in-datacenter-performance-analysis-of-a-tensor-processing-unit/).
It is a small project-specific engine, rather than an implementation of the
Google TPU architecture. The assignment still requires qmac/requant RTL and
hardware measurements; this exploration does not replace those deliverables.

## CV-Wally status

This command cannot currently execute on the unchanged CV-Wally core.
The original controller rejects its custom opcode in an RTL component
test. No tile datapath or memory/pipeline adapter is integrated.
qmac8/requant8 also still need core integration. The functional counts
below do not demonstrate Wally performance. See the
[compatibility audit](../../docs/cvw-compatibility-audit.md).

## Primary complete-model result

The final optimized TPU path retires **214,329,448** CPU instructions, compared
with unoptimized scalar B0 **9,115,503,491**. The five-stage comparison and
MAC/instruction definitions are in the [Spike overview](../../docs/spike-results.md).
The 262,234,932 result below is the original pre-hoist/pre-inline TPU variant;
its 862,447,442 comparator is an earlier qmac variant, not B0.

## Historical full-inference comparison

2026-10-26; xPack GCC 15.2.0, O2, RV64IMC/lp64, same TFLM library/model/input.
The interval includes Invoke, all 14 stage-hash checks and dispatch, and
excludes preparation/input copying/final-logit comparisons. Both variants
verify 14 hashes and every one of the 1000 final logits against Python.

| Metric | Existing B=4 qmac8/requant8 + fused GELU | TPU tile path |
|---|---:|---:|
| Core retired instructions | 862,447,442 | 262,234,932 |
| Core dynamic loads | 252,450,453 | 59,230,053 |
| Core dynamic stores | 25,142,974 | 26,616,034 |
| Core conditional branches | 62,622,320 | 28,322,169 |
| qmac8 instructions | 156,823,872 | 0 |
| requant8 instructions | 5,974,636 | 5,974,636 |
| Tile commands | 0 | 95,225 |
| Core logical read bytes | 1,709,568,065 | 146,260,753 |
| Core logical write bytes | 27,456,296 | 39,240,776 |
| Engine read bytes, descriptor included | 0 | 322,601,216 |
| Engine write bytes | 0 | 23,898,544 |
| Combined logical reads | 1,709,568,065 | 468,861,969 |
| Combined logical writes | 27,456,296 | 63,139,320 |

Retired instructions fall **69.5941%**, an instruction-count ratio of
**3.2888×**. This ratio is **not hardware speedup**. One tile command hides
many products and memory accesses from core retirement. Engine accesses above
are explicitly counted inside the plugin, not inferred to be zero from the
core load histogram. Logical bytes mean operand widths summed across executed
accesses; they are not DRAM bytes, cache-line traffic, misses or bandwidth.

The engine executes 1,255,044,864 products, including AV padding. The workload
has 1,253,683,200 mathematical MACs. Core MAC/instruction rises from about
1.454 to 4.781 because much of the computation moves into the engine. Store
instructions and total write bytes increase: the CPU writes descriptors and
the engine materializes INT32 dot tiles before CPU postprocessing. This is
a real cost, not an across-the-board improvement. All 5,974,636 requantizations
still run on the CPU; normalization, softmax, residuals, data movement and hash
checks also remain. `results/tpu/` contains exact machine-readable evidence,
code size, configuration and source/data hashes.

## Dataflow and semantics

For each at-most 8-row, 8-channel output tile, load activation rows and weight
rows into local buffers once, then reuse them for all products in that tile:

```
X [rows,K] ──► local X buffer ─┐
                              ├─► 8×8 dot array ─► INT32 tile in RAM
W [cols,K] ──► local W buffer ─┘                         │
                                                      ▼
                  CPU: bias → requant8 → optional GELU LUT → INT8 output
```

The functional plugin calculates dot products directly; it does not simulate
PE-to-PE movement or clock edges. A possible output-stationary systolic
implementation keeps one output sum per processing element (PE), streams
operands with row/column skew, and retains weights/inputs in local buffers.
The engine has 64 scalar INT8 MAC PEs compared with eight products per qmac8;
this trades additional arithmetic/storage hardware for reuse and lower CPU
control overhead. No area-equivalence claim is made.

Instruction: custom-0 `funct7=0x22`, `funct3=0`, `rd=x0`, `rs2=x0`,
`rs1=descriptor pointer`. Match `0x4400000b`, mask `0xfff07fff`. RV32 and
reserved encodings trap. The compiler wrapper has a memory clobber. Completion
is synchronous; output stores complete before subsequent CPU instructions.
There is no asynchronous DMA, queue, overlap or interrupt service within a
command in this functional model.

The 8-byte-aligned, little-endian descriptor consists of eight uint64 fields:

| Byte offset | Field | Meaning |
|---:|---|---|
| 0 | x | Signed INT8 activation pointer |
| 8 | w | Signed INT8 output-channel-major weight pointer |
| 16 | out | 4-byte-aligned dense INT32 tile output |
| 24 | k | Reduction length, 0..1024 |
| 32 | xstride | Bytes between activation rows, at least k |
| 40 | wstride | Bytes between weight rows, at least k |
| 48 | rows | 1..8 |
| 56 | cols | 1..8 |

For each r,c: `out[r*cols+c] = sum_i(int8(x[r*xstride+i]) *
int8(w[c*wstride+i])) mod 2^32`. Signed sums are interpreted as INT32.
K=0 writes zero. Input/weight pointers may be byte-unaligned. All descriptor
and operand loads occur before result stores, so overlapping sources and
outputs use snapshotted operands. Invalid descriptor shape/alignment/stride
traps as illegal instruction. Addresses are handled with RV64 unsigned
arithmetic and ordinary Spike MMU loads/stores; callers must supply valid
RAM ranges. Memory/protection faults propagate. A fault during stores may
leave a partially written destination. Restarting recomputes the tile, but is
not generally safe when those stores overwrite operands or the descriptor.
This alias/fault combination has not been tested. A hardware implementation
must forbid such overlap or define preserved state and restart behavior.
This prototype has no all-or-nothing memory commit mechanism and must not
be used on side-effectful MMIO. Privileged/coherence integration remains
hardware design work.

CPU bias addition preserves low-32-bit arithmetic; requantization and GELU
retain the original boundary exactly. The existing model's accumulator bounds
remain required. Maximum modeled local data storage is 16 KiB for two
8×1024-byte operand buffers plus 256 bytes of accumulators and the descriptor;
this is a storage-capacity budget, not synthesized SRAM area. The CPU also
uses a 256-byte temporary dot tile and a 64-byte descriptor on its stack.

## Timing assumptions: exploratory only

The plugin separately accumulates analytical counters. A hypothetical systolic
tile takes `k+rows+cols-2` compute cycles (zero for k=0), with inactive tail
PEs. Summing across this workload gives **21,271,739 modeled compute cycles**.
Adding serialized operand/descriptor reads and result writes at 16 bytes per
cycle gives **42,928,145 modeled engine cycles**:

```
sum_tiles(compute + ceil((64 + (rows+cols)*k)/16) + ceil(4*rows*cols/16))
```

These are design assumptions, not timing simulation. No setup, bank conflicts,
cache misses, bus contention, requantization, CPU execution/stalls or frequency
change is represented. No overlap is credited. In particular, dividing baseline
retired instructions by these modeled engine cycles would mix unlike quantities
and produce an invalid speedup. RTL and a defined memory system are required
for cycles, CPI, Fmax, area, energy and performance/area.

## Reproduction and checks

Prerequisites: run the repository setup/fixtures and existing full inference
once (see root README). This provides the pinned compiler, TFLM library, model,
expected logits and qmac plugin. Then:

```sh
python3 scripts/project.py tpu
# Or directly:
python3 experiments/tpu-spike/run.py
python3 experiments/tpu-spike/run.py --tests-only
```

The runner rebuilds both full-model variants, profiles both with matching
boundaries, runs the TPU host model under UBSan, and exports results/tpu.
900 cases cover M/N={1,7,8,9,17}, K={0,1,7,8,9,192,197,768,1024}, aligned/
unaligned input, GELU/no GELU and untouched output guards. An independent
INT64 scalar oracle checks every output. An extrema test checks -128×-128
accumulation. Four additional logged boundary cases exercise Spike's alternate
execution handler. Nine negative controls cover extension absence, reserved
encoding/rd, bad descriptor alignment, dimensions, K, stride and output
alignment. The existing qmac8/requant8 tests remain applicable to their
unchanged instruction implementations.

Raw logs/ELFs/disassembly and commands are under build/ and excluded from Git.
The root runner manifest records exact commands and source/data/ELF hashes.
The experiment uses the existing synthetic input and quantization: matching
the integer reference does not establish ImageNet accuracy or an improvement
in model quality. Future work: bus-backed RTL, buffer/bank design, transfer
latency, hardware cost, representative calibration and accuracy, then a fair
hardware comparison against the integrated qmac core.
