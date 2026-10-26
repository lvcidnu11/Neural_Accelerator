> Repository entry point and current status: [root README](../../README.md).
> Experiment-specific results below retain their stated measurement scope.

# Complete DeiT-Tiny integer inference — 2026-10-25

The complete non-distilled DeiT-Tiny model executes through TFLite Micro on
bare-metal RV64IMC Spike, using the original checkpoint's learned matrices:

```
INT8 image [1,3,224,224]
 -> patch embedding -> CLS/position addition
 -> 12 complete Transformer blocks:
      LN -> QKV -> QK^T/8 -> Softmax -> AV -> projection -> residual
      LN -> fc1/requant/GELU LUT (fused) -> fc2 -> residual
 -> final LN -> CLS classifier -> 1000 INT8 logits
```

This is a full model execution, not repetition of one kernel or a single block.
It uses one fixed synthetic input and a project quantization profile. It is
**not representative calibration or ImageNet accuracy validation**. The integer
profile predicts class index 38 on this input; the original FP32 model predicts
79. Thus successful integer-reference checks do not establish fidelity to FP32
classification. Accuracy/calibration remain required project work.

## What is implemented

- All patch/FC/classifier matrix weights: per-output-channel signed INT8.
- Dense activations: signed INT8 with fixed per-tensor scales; INT32 bias and
  accumulator, Q31 single-rounding requantization. Conservative per-channel
  bounds prevent INT32 overflow.
- Shared project::linear<4> uses register blocking and qmac8/requant8. fc1
  applies the precomputed GELU LUT directly before the output store, preserving
  the INT8 requantization boundary. No separate fc1 intermediate tensor exists.
- Attention Q/K/V are quantized INT8. QK uses integer dot products, then INT8
  score requantization. Softmax uses an offline Q24 exp table on score-minus-max,
  integer normalization and UINT8 probabilities at scale 1/255.
- AV encodes UINT8 probabilities as signed INT8 p-128, with the exact correction
  128*sum(V) in the accumulator bias. K=197 is padded to 200 with zero-value
  weights. Python reference independently uses direct UINT8 probability @ V.
- LayerNorm uses integer sums/squares, integer square root, Q16 INT32 affine
  constants and integer rounded division. Epsilon is converted offline into
  the integer variance domain. It is a project approximation, not a claim of
  bit-exact LiteRT builtin LayerNorm compatibility.
- Residuals rescale both operands with wide integer intermediates before
  addition and INT8 clamp. Position/CLS additions use wider integer constants.
- Matrix weights/activations are W8A8; normalization affine parameters and
  non-MAC constants use wider fixed-point metadata. Do not describe every
  learned parameter as INT8.
- No runtime floating-point arithmetic is needed by these inference kernels.
  The measured histogram contains no F/D/V/A instructions. This is functional
  ISA simulation, not physical hardware timing.

## TFLM graph representation

The exported .tflite has 14 nodes and three project custom operator types:
DEIT_EMBED_INT8, twelve DEIT_BLOCK_INT8 nodes, and DEIT_HEAD_INT8.
Each block actually computes attention, normalization, MLP and residuals.
TFLM handles graph order and tensor-arena allocation; project C++ implements
the full custom blocks. All matrix weights, integer parameters, scales and
reference stage hashes are contained in the model buffers.

This is a custom TFLM model export, not an official PyTorch converter producing
a standard builtin-only LiteRT graph. It requires model.cc registrations and
is not directly portable to an unmodified LiteRT runtime. Graph preparation
checks data types, model-data alignment and tensor quantization metadata.
The reader is specialized to the pinned model/shape; it is not a general
untrusted-model parser.

## Verification

export_model.py strictly loads the official checkpoint SHA256
a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97.
Its explicit floating forward path matches timm's original full forward.
It then independently executes the entire integer graph with NumPy INT64,
Python integer/isqrt operations and specified rounding. All twelve GELU tables
are independently checked against PyTorch FLOAT64 exact GELU for 256 inputs.

The C++ implementation passes normal host execution and UndefinedBehaviorSanitizer.
On Spike, all 14 output-stage FNV-1a hashes match Python, all 12 blocks complete,
and every one of the final 1000 INT8 logits matches exactly. Stage hashes are
diagnostic 32-bit checks, not bytewise comparison of every intermediate.
The existing optimized-kernel tests cover 640 tail/channel/alignment/fusion
combinations against a separate scalar reference on host/UBSan and Spike.

Stage checks bind this verification executable to the supplied synthetic input.
Evaluating additional inputs requires appropriate new reference hashes or a
separate evaluation build; failure on another input is not a classifier error
report. Fixed quantization scales still require representative calibration.

## Measured result

| Metric | Optimized full model |
|---|---:|
| Retired instructions, Invoke interval | 862,447,442 |
| qmac8 | 156,823,872 |
| requant8 | 5,974,636 |
| Loads | 252,450,453 |
| Stores | 25,142,974 |
| Conditional branches | 62,622,320 |
| TFLM arena actually used | 190,416 bytes |
| Reserved TFLM arena | 524,288 bytes |
| Model FlatBuffer | 6,123,312 bytes |
| Test ELF .text | 76,672 bytes |
| Test ELF .bss, including static workspace | 1,034,996 bytes |

The interval includes all 14 stage-hash checks and operator dispatch. It
excludes startup, allocation, input copying and final logit comparisons.
Mathematical MACs are 1,253,683,200; AV padding raises executed mathematical
products to 1,255,044,864. QK's output-channel tail uses the scalar fallback;
the exact qmac8 count follows from these padding/tail rules and is asserted.

No full-model baseline was measured, so there is no end-to-end speedup claim.
No cycles, CPI, Fmax, physical area or hardware speedup is measured. Static
workspace buffers and the TFLM arena are distinct allocations; arena usage
alone is not total RAM usage. The 32 MiB Spike layout has sufficient space.

## Reproduce offline

Prerequisites are the existing deit-tiny-inspect environment/checkpoint/fixture
and the tflm-linear bootstrap. run.py builds the host/RV64 TFLM libraries and
Spike plugin before inference.
For a fresh setup use `python3 scripts/project.py setup` from the repository root. No new Python packages
or global installations are required by this experiment.

```sh
cd experiments/deit-full-int8
python3 run.py
```

Or run individually:

```sh
../deit-tiny-inspect/.venv/bin/python export_model.py
python3 build.py --target host
python3 build.py --target host --sanitize
python3 build.py --target riscv
python3 profile_spike.py
```

The build reuses a pinned TFLM static library snapshot and the local xPack
GCC 15.2.0 toolchain. Model bytes are embedded with .incbin rather than millions
of C initializer values. All target buffers are static; no heap is supplied.

profile_spike.py uses debugger minstret snapshots and deterministic capped-PC-histogram
subtraction, with exact total and boundary checks. It adds no CSR instructions
to the target. Evidence is in build/profile.json, reference.json, reference.npz,
manifest.json, each build.json, host/UBSan results, and saved Spike commands,
debugger results, histograms, ELF, disassembly and map.

Next requirements: representative calibration and accuracy; comparison with
an appropriate full-model software baseline; finer operator contribution and
three-bottleneck analysis; CV-Wally integration, cycles and physical metrics.

AI assistance: Codex helped author the project integer profile, source, export,
tests and profiling. The student should review and be able to defend the choices.
