# Measured results and interpretation

All numbers below come from committed `results/reproduced/*.json`, measured
on 2026-10-25. They are functional Spike counts, not hardware cycle speedups.
See [EXPERIMENTS.md](../EXPERIMENTS.md) for ROI and metric definitions.

## Full FC1: instruction work and memory instructions

M=197, K=192, N=768; 29,048,832 mathematical MACs. The ROI is TFLM Invoke.
Parameter packing, dispatch and kernel overhead are included.

| Metric | Upstream FC | qmac8/software requant | qmac8/requant8 |
|---|---:|---:|---:|
| Retired instructions | 239,507,000 | 32,997,457 | 26,491,730 |
| Loads | 59,158,495 | 9,232,268 | 8,021,900 |
| Stores | 151,780 | 1,361,706 | 151,338 |
| Loads + stores | 59,310,275 | 10,593,974 | 8,173,238 |
| Conditional branches | 30,259,499 | 4,995,533 | 4,239,053 |
| Ordinary multiplies | 29,200,129 | 151,298 | 2 |
| qmac8 | 0 | 3,631,104 | 3,631,104 |
| requant8 | 0 | 0 | 151,296 |
| Mathematical MAC / instruction | 0.121286 | 0.880335 | 1.096525 |

Loads/stores count dynamic instructions, not bytes or cache transactions.
Ordinary multiply counts exclude the products performed inside qmac8.
The change includes wrapper/control/register-allocation effects; it is not a
claim that only scalar multiply instructions were removed.

## Incremental reductions: explicit denominators

| Scope / change | Baseline | New | Instruction reduction | Baseline / new |
|---|---:|---:|---:|---:|
| Full FC1: scalar → qmac8 | 239,507,000 | 32,997,457 | 86.22% | 7.2583× |
| Full FC1: qmac8 → qmac8/requant8 | 32,997,457 | 26,491,730 | 19.72% | 1.2456× |
| Full FC1: scalar → qmac8/requant8 | 239,507,000 | 26,491,730 | 88.94% | 9.0408× |
| Shared FC1→GELU: B1 separate → B4 separate | 26,946,684 | 17,871,511 | 33.68% | 1.5078× |
| Shared FC1→GELU: B4 separate → B4 fused | 17,871,511 | 17,228,262 | 3.60% | 1.0373× |
| Shared FC1→GELU: B1 separate → B4 fused | 26,946,684 | 17,228,262 | 36.07% | 1.5641× |

The old TFLM adapter and shared-template B=1 have different instruction counts.
Do not treat the refactor difference as an isolated fusion or blocking benefit.
Likewise, B=1 in this table is not the assignment's optimized scalar B1.

## Blocking/fusion: instruction count and arena allocation

All rows use qmac8/requant8 and return the same final 151,296 INT8 outputs.
Fused runs do not allocate/check a separate FC1 intermediate output.

| B | Fused | Invoke instructions | Arena used (bytes) |
|---|---|---:|---:|
| 1 | no | 26,946,684 | 309,584 |
| 1 | yes | 26,189,963 | 196,000 |
| 2 | no | 20,745,728 | 309,584 |
| 2 | yes | 19,913,359 | 196,000 |
| 4 | no | 17,871,511 | 309,584 |
| 4 | yes | 17,228,262 | 196,000 |

Fusion saves 113,584 arena bytes (36.69%). Allocation includes alignment,
lifetimes and interpreter metadata, so the saving need not equal one tensor's
raw 151,296 bytes. These arena figures exclude other static target buffers,
code and model constants.

## Full-model instruction mix

Includes 12 complete Transformer blocks, dispatch and 14 diagnostic stage hashes.
No full-model scalar baseline is present.

| Category | Retired instructions | Share |
|---|---:|---:|
| `integer_other` | 348,086,342 | 40.36% |
| `load` | 252,450,453 | 29.27% |
| `qmac8` | 156,823,872 | 18.18% |
| `conditional_branch` | 62,622,320 | 7.26% |
| `store` | 25,142,974 | 2.92% |
| `requant8` | 5,974,636 | 0.69% |
| `multiply` | 4,703,042 | 0.55% |
| `jump` | 3,355,479 | 0.39% |
| `divide_remainder` | 3,288,324 | 0.38% |

Total: **862,447,442**. Mathematical MAC/instruction: **1.453634**.
Combined custom-instruction share: **18.88%**.

`integer_other` is a residual instruction category, not a strict ALU count.
An instruction-category ranking is not an operator-time ranking. Full-model
operator attribution and CPU timing are still needed before claiming the next
hardware bottleneck. Attention padding/tails also mean qmac8×8 is not exactly
equal to shape-derived mathematical MACs.

## What the evidence does and does not establish

- Real custom instruction execution, independent arithmetic checks and same-output operator comparisons are verified.
- Full-model host, UBSan and Spike match 1,000 integer-reference logits plus 14 stage hashes.
- Intermediate hashes are 32-bit diagnostics; full intermediate byte comparisons are not claimed.
- Current quantized top-1 38 differs from float top-1 79 on the synthetic input. No dataset accuracy is measured.
- No hardware cycles/CPI/Fmax/physical area or full-model baseline speedup are reported.
- Standalone RTL latency and generic cell counts are in [DESIGN.md](../DESIGN.md); they do not establish CPU speedup.
