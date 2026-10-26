# Experiments and metric definitions

Run instructions are in README. Reproduce from the repository root using
`scripts/project.py`; all generated artifacts remain beneath the repository.
`setup` is separate so benchmark stages do not silently install dependencies.

## Comparison matrix

1. Standalone model-derived FC tile: M=2, K=192, N=8, 3,072 mathematical MACs.
   Same O2 C driver/data/layout; scalar, qmac8, qmac8+requant8. Counts
   22,271 / 3,199 / 2,834. Compiler is PATH RISC-V GCC 16.2.0.
2. TFLM full FC1: M=197, K=192, N=768, 29,048,832 mathematical MACs.
   Upstream FC, qmac8/software requant, qmac8/custom requant share graph,
   quantization and input. Compiler is pinned xPack GCC 15.2.0-1.
3. The same full FC1→GELU scope in all configurations. All have a LUT; this
   measures FC changes while retaining GELU. Adding GELU to a smaller execution
   range is not a GELU speedup comparison.
4. Matched templated kernel: B=1,2,4 crossed with separate/fused GELU, both
   small tile and full FC1. Compare B=1 shared kernel against B=4 shared kernel
   to isolate blocking/fusion; the old adapter is a different baseline.
5. Full DeiT-Tiny: one synthetic seed-6455 image, real pretrained weights,
   complete 12-block graph, optimized configuration only. This demonstrates
   integration/correctness and instruction mix, not full-model speedup.

## Boundaries and counters

TFLM measurements bracket `Invoke()` and include dispatch, kernel control,
loads/stores, parameter packing, requantization, LUT, and call overhead.
Startup, model conversion and final output comparisons are outside the region.
Full inference additionally includes per-node stage-hash diagnostics.
Standalone tile boundaries are `kernel_begin` inclusive to `kernel_end`
exclusive; its verification stores are deliberately included in every variant.

Spike debugger reads `minstret` at boundary labels; target code uses no CSR
instructions. The profiler subtracts capped mnemonic histograms before/after
the region and asserts their sum equals the counter delta. On the validated
Spike version `--instructions=N` executes N+1 instructions, so capped profiling
uses the measured counter minus one. This behavior is version-sensitive; keep
all assertions enabled when changing the simulator.

- Retired instructions: dynamic completed ISA instructions, including each
  compressed instruction once and each custom instruction once.
- Instruction-count ratio: baseline/custom retired count; not cycle speedup.
- Dynamic memory instructions: retired loads plus stores; not bytes, transactions,
  cache misses, or DRAM traffic. Wider loads do not automatically reduce bytes.
- MAC/instruction: shape-derived mathematical MACs divided by retired count.
  Full-model attention padding is reported separately from mathematical work.
- Custom utilization: qmac8/requant8 retirements divided by total retirements.
- Code size: linked ELF `.text`; constants/model `.rodata` and static `.bss`
  are separate from TFLM arena usage. Arena alone is not total target RAM.
- `integer_other` is a residual mnemonic category, not a strict ALU-only count.

No warmup or cache study is claimed: Spike is a functional simulator and the
fixed executable/input instruction counts are deterministic. No hardware
cycles, CPI, Fmax, power, SAED area or speedup/area figures are inferred.

## Correctness and limitations

ISA tests use independent Python arbitrary precision arithmetic and directed
boundary, random, overlapping-register/x0, signed-lane, wrap and trap cases.
There are 1,645 qmac vectors plus 33 chain executions, 2,708 unique requant ISA
vectors plus 308 paired software checks, and 20 kernel tail/alignment cases per
custom variant. Blocking/fusion tests cover 640 shape/alignment/configuration
combinations, with independent scalar reference and canaries, on host UBSan
and actual Spike. GELU tables compare all 256 possible inputs against an
independent PyTorch float64 implementation.

Full inference compares all final 1,000 logits with the NumPy integer reference
and checks 14 intermediate 32-bit hashes. Hashes are diagnostics, not proof of
byte-for-byte equality of every intermediate. Both reference and target follow
project integer LayerNorm/Softmax rules. A match establishes implementation
consistency, not dataset accuracy or generic LiteRT numerical equivalence.
Model quantization on one synthetic input is insufficient for accuracy claims.

## Provenance

TFLM revision: `8f6596d99898deaac36ebd8ff8bf1b9754fd2cd1` (unmodified upstream).
Python direct package versions and download/toolchain hashes are pinned in
`requirements.txt`, `requirements-lock.txt` and `bootstrap.py`. Generated build manifests record compiler
versions, exact flags/commands, model/input/source hashes and simulator details.
The official checkpoint is linked by `download_weights.py`; full SHA256:
`a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97`.

Export regenerated summaries with `python3 scripts/publish_results.py` after
running all stages. Committed reference results identify their original run
scope and provenance; relocation verification is recorded separately.
