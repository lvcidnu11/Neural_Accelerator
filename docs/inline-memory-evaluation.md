# Is inlining worth the memory cost?

## Decision

For this exact workload and current memory layout, retaining optional inlining
is a reasonable tradeoff. The measured code increase is 152 to 416 bytes.
The qmac path has no observed stack increase, while the TPU path uses more
stack. A tight memory target should apply TPU parameter hoisting first and
leave inlining optional. Physical speedup still needs CV-Wally measurement.

The previous 18.27% TPU reduction combines hoisting and inlining. The marginal
benefit of inlining after hoisting is 2.68%, not 18.27%.

## Matched marginal comparison

| Configuration pair | Extra .text | Code growth | Fewer retired instructions | Extra observed stack writes |
|---|---:|---:|---:|---:|
| qmac control vs qmac inline | 152 B | 0.198% | 3.7315% | 0 B |
| TPU control vs TPU inline | 416 B | 0.547% | 2.2046% | 248 B |
| TPU hoist vs TPU hoist + inline | 392 B | 0.515% | 2.6812% | 200 B |

All variants keep .rodata=6,288,208 B, .data=2,920 B and .bss=1,034,996 B.
The tensor arena actually used remains 190,416 B, and the linker reserves
64 KiB of stack headroom in every build. The present linker puts .text in
RAM, so added instructions consume RAM address space here even though writable
data is unchanged. On a ROM/flash execution design the placement would differ.
The full-model code segment is about 75 KiB, so reporting growth only as a
fraction of the multi-megabyte model would hide the relevant instruction cost.

## Observed stack footprint

| Variant | Deepest modified stack word, from stack top |
|---|---:|
| qmac control | 1,112 B |
| qmac inline | 1,112 B |
| TPU control | 1,392 B |
| TPU inline | 1,640 B |
| TPU hoist | 1,416 B |
| TPU hoist + inline | 1,616 B |

The audit paints the reserved 64 KiB before any C call and scans it using
register-only assembly after main returns. Results have 8-byte resolution and
cover initialization, inference and verification for the same synthetic input.
The added instrumentation preserves the original inference instruction counts
and all 14 stage checks / 1000 final logits. This detects written stack depth,
not unwritten frame reservation or worst-case stack use across arbitrary inputs.
Interrupt/exception stack requirements are outside this bare-metal fixture.

GCC -fstack-usage reports per-function frames. The audit recompiles the three
application translation units and verifies each resulting object is byte-for-byte
identical to the original object. Thus diagnostics do not change the measured
kernels. These per-function reports alone cannot establish whole-program peak
stack usage, especially through the TFLM library's indirect calls.

## Functional instruction-cache sensitivity

Spike's configured models have four ways and 64-byte lines. Before/after capped
runs subtract the same initialization prefix, preserving the cache state warmed
by that prefix. Each cache-access total equals the corresponding retired-counter
limit. Results below are misses inside the complete inference ROI.

| Variant | 4 KiB I-cache misses | 16 KiB I-cache misses |
|---|---:|---:|
| qmac control | 385 | 77 |
| qmac inline | 292 | 77 |
| TPU hoist | 181 | 64 |
| TPU hoist + inline | 322 | 79 |

On TPU, inlining after hoisting saves 5,904,946 CPU instructions but adds 141
modeled misses at 4 KiB, or 15 at 16 KiB. As a sensitivity calculation only,
C = I + P*M assumes one cycle per CPU instruction plus P extra cycles per miss,
and holds other costs constant. At 4 KiB, the extra misses cancel that savings
only at P about 41,879 cycles per added miss. This is not a hardware prediction.
Actual pipelines, data-cache behavior, fetch stalls and overlap need RTL.

The cache's default replacement uses an LFSR. The configured functional model
looks up instruction start addresses and does not represent CV-Wally fetch
width or cross-line instruction behavior. The installed Spike MMU passes an
end address into the trace size field, causing its Bytes Read counter to be
address-inflated. The audit excludes that byte counter entirely. Address-based
read-access and miss counts are the only cache statistics used here.

## Choosing a configuration

For TPU, hoisting alone reduces instructions by 16.02% with only 44 extra
code bytes and 24 extra observed stack bytes relative to the original TPU.
It also uses 372 fewer code bytes, 224 fewer observed stack bytes and fewer
instructions than the inline-only TPU configuration. Thus it is the stronger
first choice on these dimensions.

Adding inline after hoisting costs another 392 code bytes and 200 observed
stack bytes for 2.68% fewer instructions. That is attractive with the current
64 KiB stack reservation, but a target with a small stack budget may prefer
hoisting alone. Keep both options for hardware evaluation and use a defined
stack safety margin rather than the canary measurement alone.

## Reproduce

First run the existing six-way software evaluation to retain its ELFs, object
files and exact build logs. Then point this audit at that evaluation build:

```sh
python3 experiments/inline-memory-2026-11-02/audit.py --eval experiments/tpu-evaluation-2026-11-02/build
python3 experiments/inline-memory-2026-11-02/report.py
```

Source references:
- GCC inline: https://gcc.gnu.org/onlinedocs/gcc/Inline.html
- GCC stack diagnostics: https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html#index-Wstack-usage
- Spike cache model: https://github.com/riscv-software-src/riscv-isa-sim/blob/master/riscv/cachesim.cc
- Local /opt/homebrew/include/riscv/mmu.h and cachesim.h for installed tracer details.
- The project linker: experiments/tflm-linear/link.ld.

AI assistance: Codex assisted with the memory audit, cache sensitivity study,
interpretation and updated slides. Stack and cache measurements do not establish
hardware speedup, worst-case memory requirements, or a dataset accuracy result.
