> Repository entry point and current status: [root README](../../README.md).
> Experiment-specific results below retain their stated measurement scope.

# Incremental evaluation: qmac8 plus custom requant8 in Spike

Verified 2026-10-24. Previous qmac8-only experiment is preserved in
../qmac8-spike. This experiment adds an actual requant8 decode/handler, uses
it from C, and evaluates three configurations on identical model-derived data.
It is not a model-calibration/accuracy result, full-workload evaluation, GELU
implementation or CV-Wally cycle-performance result.

## Reproduce

Use the existing model environment and fixtures from ../deit-tiny-inspect and
../deit-linear-int8; installed RISC-V GCC/binutils and Spike with matching C++
development headers; host C++17 compiler; Python. Tested on macOS arm64.

```sh
cd experiments/qmac8-requant-spike
python3 run.py
```

If generated parent vectors or the previous comparison are absent, run.py
builds them using their saved scripts. Initial model weights/environment must
already be prepared. No network is needed after those prerequisites.
Optional variables: RISCV_CC, RISCV_SIM, CXX, SPIKE_PREFIX, MODEL_PYTHON.
The library still registers extension name xqmac8 but includes both handlers.
Do not load the old qmac8-only library at the same time.

Manual execution of the combined target on this machine:

```sh
spike --isa=RV64IMC -m32 \
  --extlib="$PWD/build/libqmac8.dylib" --extension=xqmac8 \
  build/qmac8_requant/program.elf
```

## Same workload, incremental extensions

DeiT-Tiny blocks.0.mlp.fc1 tile M=2,K=192,N=8; actual pretrained weights,
synthetic captured input; 16 outputs, 3,072 MACs. Same input/weights/bias,
scales, zero points=0, single-rounding rule, static layout, accumulator/output
verification stores and driver in all configurations. The scalar baseline is
O2, not a debug O0 comparison. No LTO; fno-inline; all explicit asm wrappers
are always_inline so they emit instructions rather than function calls.

Scalar and qmac8 counts exactly reproduce the preserved prior experiment.
The combined build changes only the software requantization path to the new
instruction in the same parameterized kernel.c.

| Measured metric | Scalar | qmac8 | qmac8 + requant8 |
|---|---:|---:|---:|
| Retired instructions | 22,271 | 3,199 | 2,834 |
| Mathematical MACs | 3,072 | 3,072 | 3,072 |
| qmac8 executions | 0 | 384 | 384 |
| requant8 executions | 0 | 0 | 16 |
| MAC / instruction | 0.137937 | 0.960300 | 1.083980 |
| Load instructions | 6,235 | 869 | 846 |
| Store instructions | 61 | 60 | 48 |
| Conditional branches | 3,156 | 500 | 452 |
| Ordinary multiplication instructions | 3,088 | 16 | 0 |
| Load+store instructions / MAC | 2.049479 | 0.302409 | 0.291016 |
| Full target .text bytes, including driver/tests | 724 | 766 | 748 |

Incremental reduction versus qmac8: (3,199-2,834)/3,199 = 11.41%, with
instruction-count ratio 1.129x. Combined reduction versus scalar: 87.27%,
ratio 7.859x. These are instruction-count ratios, NOT cycle speedups or
end-to-end model speedups.

The software requant helpers occupied 343 instructions in the previous trace,
excluding call-site setup. The new path also removes their call overhead and
changes register allocation/prologue requirements, while adding explicit
parameter packing. The net reduction of 365 includes all these effects; it is
not a claim that the 343 helper instructions alone became negative overhead.

## Parameter preparation is included

requant8 accepts accumulator in rs1 and packed multiplier/shift/output-zero-point
in rs2. Each output uses its channel-specific multiplier/shift. Runtime loads,
shift/OR packing and the requant8 instruction itself are inside the measured
region. No prepacked-parameter table or hidden Spike configuration removes that
cost. Output zero-point is fixed to zero for this workload; nonzero cases are
verified independently. One requant8 produces one INT8 output, so 16 execute.

Both input/weight arrays have the same 8-byte alignment in all builds. qmac8
uses aligned alias-safe builtin copies, scalar fallback for unaligned row/
channel pointers, and a scalar remainder loop. Operand data bytes are not
automatically reduced by using wider loads.

## Verification

- All original qmac8 tests pass: 1,645 ISA cases plus 33 dependent qmac8s,
  with signed lanes, register overlap, x0, wrapping and full-register results.
- 2,708 requant8 ISA cases: 308 inherited software-reference cases, 864 directed
  parameter/extreme/zero-point combinations, 1,024 random cases, 512 explicit
  overlap/x0 cases. Independent Python arbitrary-precision reference.
- The inherited 308 cases ALSO execute paired software/custom checks, giving
  3,016 actual requant8 retirements in the test trace. Distinct ISA vector count
  remains 2,708; do not call the repeated paired cases additional unique tests.
- Tests cover ignored accumulator high bits, output sign extension, negative
  ties, saturation, zero multiplier, shifts -31 through +30, INT32 endpoints,
  full valid nonnegative multiplier range and signed output zero-points.
- fast and commit-logged handlers pass for both instruction test executables.
- Twenty tail/alignment kernel cases pass in EACH custom configuration.
- Every model-tile run checks all accumulator/INT8 outputs and 308 software
  requant reference cases; all three configurations pass fast/logged execution.
- Six requant negative controls: reserved encoding, negative multiplier,
  reserved parameter bits, shift -32, shift +31, and missing extension.
  Original qmac missing-extension/reserved-encoding controls also pass.
- Negative tests use bounded execution and explicitly check illegal-instruction
  diagnostics rather than interpreting process exit alone as failure evidence.

## Measurement boundary and limitations

Only commit records between runtime kernel_begin (inclusive) and kernel_end
(exclusive) count. Includes argument setup, call/return, function overhead,
bias, multiplication, packing, requantization and verification accumulator/output
stores. Excludes boot/startup, the 308 directed checks, output comparison and
host termination/spins. Uses the same extension-loaded simulator in all measured
runs; scalar additionally passes on stock Spike. Compressed instructions retire
once, and functional custom handlers do not imply target cycle timing.

Scaled output is mathematically computed in wide precision before final INT8
clamp. On the stated valid domain of the original software helper it is exactly
equivalent. The instruction additionally defines saturation for large scaled
values outside that helper's INT32-output precondition; these cases use an
independent wide reference. See ISA.md for complete parameter/exception rules.

No cycle counts, CPI, cache misses, Fmax, physical area, accuracy or full-model
speedup were measured. Existing one/two/eight-lane standalone RTL candidates
are not CPU integrations. More representative operators and full workload
remain necessary, and calibration still uses only a synthetic fixture.

## Evidence

- ISA.md: encoding, packed parameter fields, exact arithmetic and exceptions.
- extension.cc: registered qmac8 and requant8 handlers/disassemblers.
- requant8.h: packed-parameter builder and real .insn C wrappers.
- kernel.c: scalar/qmac8/combined configurations; no GELU implementation.
- generate_requant_tests.py and requant_tests.c: reference and actual ISA tests.
- run.py / profile.py: full reproducible build/test/measurement pipeline.
- build/comparison.json: measured counts, fractions and all mnemonic mixes.
- build/{scalar,qmac8,qmac8_requant}/: ELF, map, disassembly, symbols and trace.
- build/requant-tests/: fast/logged test execution; test vectors/manifests in build.
- build/manifest.json: exact commands, source/generated fixture hashes, parent
  provenance, matching installed Spike-header hashes, simulator/library hashes.

Sources now live in the course project repository; generated outputs are ignored.
No GitHub commit/push is performed by this experiment. These files were created
with AI assistance; understand the semantics/tests and follow course policy.
