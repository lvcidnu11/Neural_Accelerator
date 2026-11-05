# Full-inference incremental comparison

Same 12-block DeiT-Tiny model, quantization, input and 14 stage checks.
GCC 15.2.0 -O2, RV64IMC, functional Spike.

| Stage | Configuration | Retired instructions | Useful MAC / CPU instruction | Change from previous |
|---|---|---:|---:|---:|
| B0 | Scalar B=1, software requant, separate GELU LUT; no Linear inline | 9,115,503,491 | 0.1375 | — |
| B1 | Scalar B=4, fused GELU LUT, Linear inline | 6,095,818,622 | 0.2057 | 33.13% fewer |
| X1 | B1 + qmac8; software requant | 890,189,414 | 1.4083 | 85.40% fewer |
| X2 | X1 + requant8 | 830,264,926 | 1.5100 | 6.73% fewer |
| TPU | Tile dot command + requant8; CPU parameter reuse + inline | 214,329,448 | 5.8493 | 74.19% fewer |

Useful mathematical model MACs: 1,253,683,200. MAC/instruction divides this fixed numerator by CPU retired instructions; TPU work is offloaded. These are not cycle speedups.
B0 is the 9.116-billion scalar baseline; B1 is its software-optimized 6.096-billion stage. See [the complete Spike overview](spike-results.md) for the supporting ablations.

All five runs use input SHA-256 `3325824ca88991b5714d63f46474418805015cd2bce9cea8ec6e3bef83820ecc` and model SHA-256 `958de3ce3f1b421a6de6c25d2c132f46ae09d1728b000586a7e0848118a6ed01`.
All pass 14 stage hashes and all 1,000 final logits. B0 passes stock Spike.

B0 uses a noinline standalone GELU LUT function and an intermediate tensor inside the existing custom block node. It does not add a separate TFLM operator dispatch. Its W8A8 GELU still uses the same lookup table; a non-LUT baseline requires a separate implementation and correctness evaluation. B1 bundles blocking, fusion and inline, so B0→B1 is their combined effect, not an isolated measurement. X1→X2 also includes compiler/code-generation changes. TPU changes the execution engine and adds CPU parameter reuse; it is a functional Spike experiment.

Reproduce all stages after setup with `python3 scripts/project.py checkpoint`. To inspect committed results without rerunning, use `python3 experiments/b0-full-inference-2026-11-03/collect.py --saved`. Source overlays, compiler flags and full logs are saved in the respective build directories. Original project sources are not overwritten.

These are CPU retired-instruction counts, not CV-Wally cycles or hardware speedups. Dataset accuracy, hardware timing and physical area are not established by this Spike comparison.
