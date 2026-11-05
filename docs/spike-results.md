# Spike results overview

Updated 2026-11-06 from committed measurements (2026-10-25 to 2026-11-03).
This update recalculates derived metrics and documentation; it does not claim a new full-model run.

## Complete DeiT-Tiny: primary comparison

**B0 is 9,115,503,491 instructions (9.116 billion). The final TPU path is 214,329,448 (0.214 billion).**
B1 at 6.096 billion is the software-optimized scalar stage, not the unoptimized B0 baseline.
The earlier 862-million qmac and 262-million TPU counts belong to pre-inline/pre-hoist variants below.

Every row executes the same complete 12-block model and the same synthetic input.
Each checks 14 stage hashes and all 1,000 final integer logits.
The measured interval includes operator dispatch and the stage-hash checks.

| Stage | Configuration | CPU retired instructions | Useful MAC / CPU instruction | Reduction from previous | Count ratio vs B0 |
|---|---|---:|---:|---:|---:|
| B0 | Scalar B=1; separate GELU LUT; software requant; no inline | 9,115,503,491 | 0.1375 | — | 1.00x |
| B1 | Scalar B=4; fused GELU LUT; Linear inline | 6,095,818,622 | 0.2057 | 33.13% | 1.50x |
| X1 | B1 + qmac8; software requant | 890,189,414 | 1.4083 | 85.40% | 10.24x |
| X2 | X1 + requant8 | 830,264,926 | 1.5100 | 6.73% | 10.98x |
| TPU | Tile offload + requant8; parameter reuse + inline | 214,329,448 | 5.8493 | 74.19% | 42.53x |

The full model contains **1,253,683,200 useful mathematical MACs**.
MAC/instruction is this fixed numerator divided by each row's CPU retired count.
For TPU it includes accelerator work, so it is not the number of MACs executed directly by the CPU per instruction.
One tile command hides many operations and memory accesses from CPU retirement.
The final count ratio is 42.53x versus B0; this is **not a measured hardware speedup**.

B0 uses the same INT8 GELU LUT as the optimized paths, in a separate noinline function inside
the existing block node. B0 is not upstream TFLM full-model inference or a non-LUT GELU implementation.
B0 to B1 combines B=4 blocking, FC/GELU fusion and inline. X1 to X2 also includes compiler code-generation effects.
The final TPU row includes the CPU parameter-reuse and inline improvements.

## Full-model software ablations

These full-inference variants use the same mathematical MAC numerator and input/reference.
Compare each inline/hoist variant against its own original execution path.

| Configuration | CPU retired instructions | Useful MAC / CPU instruction | ELF .text bytes |
|---|---:|---:|---:|
| qmac-control | 862,447,442 | 1.4536 | 76,672 |
| qmac-inline | 830,264,926 | 1.5100 | 76,824 |
| tpu-control | 262,234,932 | 4.7808 | 76,104 |
| tpu-inline | 256,453,645 | 4.8885 | 76,520 |
| tpu-hoist | 220,234,394 | 5.6925 | 76,148 |
| tpu-inline-hoist | 214,329,448 | 5.8493 | 76,540 |

The original qmac path is 862,447,442; inline reduces it to the X2 value of 830,264,926.
The original TPU path is 262,234,932; hoist + inline reduces it to the final 214,329,448.
These are supporting ablations, not replacement B0 baselines.

## FC1-only instruction experiments

FC1 has M=197, K=192, N=768: **29,048,832 MACs**. The interval is TFLM Invoke().
These counts cover a single operator, not the complete model.

| FC1 implementation | CPU retired instructions | MAC / CPU instruction |
|---|---:|---:|
| TFLM upstream FC | 239,507,000 | 0.1213 |
| TFLM qmac8 + software requant | 32,997,457 | 0.8803 |
| TFLM qmac8 + requant8 | 26,491,730 | 1.0965 |

For the next study, every row uses the same shared qmac/requant kernel and includes the same GELU LUT.
Its B=1 row is a packed-kernel specialization, **not scalar stage B1**.

| Channels per block | GELU fusion | CPU retired instructions | MAC / CPU instruction | Arena bytes |
|---|---|---:|---:|---:|
| 1 | separate | 26,946,684 | 1.0780 | 309,584 |
| 1 | fused | 26,189,963 | 1.1092 | 196,000 |
| 2 | separate | 20,745,728 | 1.4002 | 309,584 |
| 2 | fused | 19,913,359 | 1.4588 | 196,000 |
| 4 | separate | 17,871,511 | 1.6254 | 309,584 |
| 4 | fused | 17,228,262 | 1.6861 | 196,000 |

Do not multiply the FC1, blocking/fusion and full-model count ratios together.

## TPU traffic and correctness boundaries

The final Spike TPU variant executes **95,225 tile commands** and **1,255,044,864 padded products**.
Its engine performs 322,601,216 logical read bytes and 23,898,544 logical write bytes,
reported separately from CPU loads/stores. The padded product count is not the useful-MAC numerator above.
Logical bytes are not cache/DRAM traffic. Analytical engine timing counters are not included as measurements.

All five full-model paths match the same project integer reference. This establishes tested arithmetic
consistency, not calibrated dataset accuracy. One synthetic image and synthetic activation scales
cannot establish ImageNet top-1/top-5 accuracy. No hardware cycles, Fmax or physical area are reported here.

## Evidence and reproduction

- [Machine-readable summary](../results/spike-summary.json) contains relative evidence paths and SHA-256 hashes.
- [Five-stage source measurements](../results/incremental-inference-2026-11-03/) and [B1/X2/TPU profiles](../results/full-inference-2026-11-03/).
- [Software ablations](../results/software-2026-11-02/comparison.json), [FC1](../results/reproduced/fc1.json), [blocking/fusion](../results/reproduced/blocking-fusion.json).

Regenerate/check the summary from committed data:

```sh
python3 scripts/summarize_spike_results.py
python3 scripts/summarize_spike_results.py --check
```

Rerun the actual measurements after installing the pinned dependencies:

```sh
python3 scripts/project.py setup
python3 scripts/project.py checkpoint
```

The summary command does not execute Spike. The checkpoint command executes the workloads;
fresh counts must be verified against reference outputs and their tool/source configuration.
