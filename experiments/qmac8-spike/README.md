> Repository entry point and current status: [root README](../../README.md).
> Experiment-specific results below retain their stated measurement scope.

# Actual qmac8 execution in Spike and Linear-tile comparison

Verified 2026-10-24. This experiment implements Xqmac8 as a loadable Spike
extension, issues the encoding from C, verifies real instruction execution,
and compares an optimized scalar Linear tile with a qmac8 version.
The standard Spike installation is unchanged. This is an ISA-level result,
not CV-Wally cycle performance or a calibrated full-model speedup.

## Reproduce

Requires the earlier deit-linear-int8 fixture and its Python environment,
RISC-V GCC/binutils, installed Spike with matching development headers, host
C++17 compiler, and Python. Current environment is macOS arm64. The script
also selects shared-library flags for Linux, but that platform is untested.

```sh
cd experiments/qmac8-spike
python3 run.py
```

If parent vectors are absent, run.py invokes the parent baseline run script.
The prerequisite float fixture and model environment must already be prepared
as described in ../deit-tiny-inspect and ../deit-linear-int8. No network is
needed for this experiment after those dependencies are ready.

Optional environment variables: RISCV_CC, RISCV_SIM, CXX, SPIKE_PREFIX and
MODEL_PYTHON. SPIKE_PREFIX defaults to the real executable's installation
prefix (following symlinks), not a separate Homebrew API lookup. For bitwise
reproduction use the simulator/header/compiler hashes and exact commands in
build/manifest.json.

On this machine a manual custom run has this form:

```sh
spike --isa=RV64IMC -m32 \
  --extlib="$PWD/build/libqmac8.dylib" --extension=xqmac8 \
  build/qmac8/program.elf
```

The built .dylib/.so adds the instruction handler and disassembly to Spike.
Without that extension the encoding traps as illegal. Stock objdump may show
`.insn 4, 0x...`; Spike's extension-aware trace displays `qmac8 rd,rs1,rs2`.

## Compared workload and controls

- Actual DeiT-Tiny blocks.0.mlp.fc1 weights, captured synthetic activations.
- Tile M=2,K=192,N=8: 16 outputs, 3,072 mathematical MACs.
- Input/weight zero points=0; same per-channel scales/multipliers, bias,
  single-rounding requantization and Python integer-reference outputs.
- Same main.c, aligned input/header, startup, linker, RV64IMC/lp64/medany,
  -O2 -fno-inline, no LTO, and linker relaxation disabled.
- Same source kernel.c with only USE_QMAC8 selecting the packed loop.
- Same extension-loaded Spike executable in both measured runs. The scalar
  ELF is also run on stock Spike to verify it has no custom dependency.
- Every target run checks accumulators, INT8 outputs and 308 requantization
  cases. Both fast and logged execution modes pass.
- Calibration is still synthetic-fixture-only. This is one small operator tile,
  not full operator/model evaluation, model accuracy or a full-workload result.

## Measured ISA-level results

| Metric | Scalar | qmac8 |
|---|---:|---:|
| Retired instructions in marked computation interval | 22,271 | 3,199 |
| Mathematical MACs | 3,072 | 3,072 |
| MAC / retired instruction | 0.137937 | 0.960300 |
| Load instructions, including stack/parameters | 6,235 | 869 |
| Store instructions, including stack/verification | 61 | 60 |
| Conditional branches | 3,156 | 500 |
| Ordinary multiply instructions | 3,088 | 16 |
| Custom qmac8 instructions | 0 | 384 |
| Custom-instruction fraction | 0 | 12.004% |
| Load+store instructions / MAC | 2.049479 | 0.302409 |
| Full ELF .text bytes (includes driver/test code) | 724 | 766 |

Instruction-count ratio = 22,271 / 3,199 = 6.962x. Reduction = 85.64%.
These are NOT cycle speedup, CPI, timing, area or clock-frequency measurements.
The baseline is the existing O2 scalar implementation, not a claim of the best
possible software optimization. Do not generalize this tile to a full-model
speedup or claim the model-level course performance target is already met.

For 16 outputs, each needs 192/8=24 qmac8, giving 384 executions. The packed
steady loop is two 64-bit loads, qmac8, two pointer increments and one branch:
6 retired instructions per eight MACs. Scalar takes 7 per MAC in its inner loop.
The final benchmark includes all remaining function, parameter, alignment,
tail-path, output and requantization overhead as well.

Operand-data loads change from 6,144 byte loads to 768 eight-byte loads.
Both read 6,144 bytes of operand data for this simple kernel. Fewer load
instructions do not establish reduced DRAM traffic or cache miss rates.
The remaining 16 ordinary mul instructions belong to software requantization.

The accumulator is kept as the architectural 64-bit qmac8 result between
iterations. Casting through INT32 on every iteration would make GCC insert
redundant sign extensions, even though the ISA already guarantees them. It is
converted back once before the scalar tail/requantization. All tests were rerun
after this implementation change.

## Packing, alignment and tail behavior

The input/header arrays have explicit 8-byte alignment in BOTH builds. The
packed loop checks each row/channel pointer. Only if both are aligned does it
use compiler builtin memcpy from an assumed-aligned source, allowing an ld
without violating C aliasing. K=192 guarantees all model-tile rows and full
groups stay aligned. No offline repacking is hidden in this comparison.

Unaligned row/channel inputs use the scalar path. Fewer than eight remaining
elements use the scalar tail. The instruction itself accesses only registers
and does not perform memory operations or implicit loads.

## Verification

- 1,645 actual ISA cases: 1,133 normal directed/random cases plus 512 explicit
  overlap/x0 cases. Covers signed lanes, extrema, lane mapping, arbitrary old
  rd high bits, INT32 wrap and complete 64-bit sign extension.
- 33 dependent qmac8 executions test chained accumulation across wrap.
- Total qmac8 executions in the ISA test trace: 1,678, verified from encodings.
- All ISA tests pass fast and logged handlers; handler source snapshots all
  operands before writeback to handle overlap correctly.
- 20 C-kernel cases: K=0,1,7,8,9,15,16,17,31,32 with aligned/offset pointers,
  multiple rows/channels, negative INT8 inputs, bias and requantization.
- Model tile: all 16 INT32 accumulators and INT8 outputs match the independent
  parent Python reference in both variants; 308 requantization checks pass.
- Negative controls: no extension and a reserved funct3 encoding each report
  trap_illegal_instruction. Runs are bounded by --instructions; the diagnostic
  is checked explicitly rather than treating the bounded process exit as pass.

The profiling boundary is runtime kernel_begin inclusive to kernel_end
exclusive and includes helper callees, argument setup, call/return, bias,
requantization and both accumulator/output stores. The accumulator stores are
verification instrumentation held equal in both variants. Boot ROM, startup,
the 308 correctness cases, output comparisons, tohost and hang are excluded.
Only commit records are counted; trace lines are not instruction counts.

## Files

- ISA.md: instruction encoding, signed lanes, accumulator/output rules, overlap.
- extension.cc: actual Spike instruction, decode registration and disassembler.
- qmac8.h: inline .insn wrappers, no GCC backend change required.
- kernel.c / kernel.h: scalar/qmac8 variants and shared requantization.
- main.c: identical benchmark driver and reference checks.
- generate_tests.py, instruction_tests.c, tail_tests.c: independent ISA/kernel tests.
- run.py: extension/target builds, test runs, negative controls and profiling.
- profile.py: trace counting, instruction-mix and code-size comparison.
- build/comparison.json: actual reported measurements.
- build/manifest.json: exact commands, source/generated-data/header/binary hashes.
- build/{scalar,qmac8}/: ELF, map, disassembly, symbols, trace, section sizes.
- build/instruction-tests/: actual ISA executable and logged trace.
- build/tail-tests/: actual tail/alignment executable.

## Remaining midterm work

### Remaining instruction attribution (2026-10-24)

`python3 analyze_remaining.py` attributes the existing marked qmac8 trace:

| Region | Retired instructions | Fraction |
|---|---:|---:|
| Packed loop: 384 iterations x 6 instructions | 2,304 | 72.02% |
| multiply_quantized helper | 231 | 7.22% |
| clamp_int8 helper | 112 | 3.50% |
| Remaining setup/outer control/memory/call overhead | 552 | 17.26% |
| Total | 3,199 | 100% |

The two requantization helpers total 343 instructions (10.72%). Their returns
are counted in the helper region, but call-site setup and call instructions
remain in the other group. These percentages are instruction shares, not
execution-time shares or a demonstrated CPU bottleneck ranking. The loop still
dominates instruction count; a custom requantization instruction is a required
separate design/evaluation step, not a promise of another qmac-sized benefit.
No requantization custom instruction has been implemented by this analysis.

The future instruction would implement the same scaling, rounding and INT8
clamping rules as the existing software reference. It needs its own semantics,
encoding, Spike handler and tests. qmac8 itself continues to output INT32.
Arithmetic parameters and exact rounding cannot be changed just to improve
the benchmark result. Detailed attribution and evidence hashes are saved in
build/remaining-analysis.json.

Current progress covers an actual MAC custom instruction and short verification
tests, plus one model-derived tile comparison. It does not complete the project.
Custom requantization, more representative operators/full workload, model
calibration/accuracy, repository-level packaging and slides remain outstanding.
The third source read, RTL integration and physical performance also remain
unresolved. Prior standalone RTL unit results are not CPU pipeline results.

These files were produced with AI assistance. Review the semantics, code and
evidence and follow course policy before submission.
