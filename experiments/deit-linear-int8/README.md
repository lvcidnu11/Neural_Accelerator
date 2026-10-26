> Repository entry point and current status: [root README](../../README.md).
> Experiment-specific results below retain their stated measurement scope.

# Model-derived W8A8 Linear: integer reference and scalar Spike tile

Verified 2026-10-24. This is operator verification using the first DeiT-Tiny
block's MLP fc1 weights and synthetic captured activations. The user confirmed
there is no calibration image dataset yet. This is NOT calibrated W8A8 model
inference or ImageNet accuracy evidence, and it implements no custom ISA.

## Scope

- Source model: official non-distilled deit_tiny_patch16_224, strictly loaded
  and hashed by ../deit-tiny-inspect. Operator: blocks.0.mlp.fc1.
- Original operator X[197,192] * W[768,192]^T + bias[768] -> Y[197,768].
- Python computes an integer reference for the full operator on the host.
- C and Spike execute only input rows 0..1 and output channels 0..7:
  M=2, K=192, N=8, 16 outputs and 3,072 mathematical MACs. K is not shortened.
- FP32 weights are the actual pretrained weights; the captured activation
  values come from the prior seed-6455 synthetic full-model input.

## Explicit provisional quantization profile

1. All zero points = 0. Input and weights quantized symmetrically to [-127,127].
2. Input scale sx = max(abs(full input fixture))/127, per tensor.
3. Weight scale sw[c] = max(abs(weight row c))/127, per output channel.
4. Output scale sy = max(abs(full FP32 output fixture))/127, per tensor.
5. Offline quantization rounds nearest with ties away from zero.
6. bias_q[c] = round(bias_fp[c]/(sx*sw[c])), stored in INT32.
7. acc = bias_q[c] + sum(input_q[row,i]*weight_q[c,i]), INT32.
8. Effective multiplier sx*sw[c]/sy is represented by Q31 multiplier m[c]
   and exponent shift[c], following QuantizeMultiplier's frexp convention.
9. Requantization explicitly selects TFLite v2.19.0 SINGLE-rounding scalar
   int32 arithmetic (TFLITE_SINGLE_ROUNDING=1):

```text
bits = 31 - shift
scaled = floor((acc * multiplier + 2^(bits-1)) / 2^bits)
output = clamp(scaled, -128, 127)
```

Do not silently substitute the double-rounding build or ties-away rounding
for this runtime formula: exact boundary results can differ. Offline float
quantization rounding and runtime requantization are separately specified.
Only the selected scalar arithmetic formula is used here; no TFLite model
conversion or claim of complete LiteRT operator/runtime compatibility is made.

All scales and multiplier parameters are calculated offline. The C target
contains integer data/arithmetic only, uses static allocation, and is compiled
for RV64IMC/lp64. Unlike the earlier wrap-semantic custom-unit experiment, this
baseline does not rely on overflow: for every possible input with |q|<=127,
the conservative bound `127*sum(abs(w_q))+abs(bias_q)` over every channel is
at most 1,069,197. This bounds all intermediate partial accumulators well inside
INT32. Requantization uses INT64 intermediates with validated input domains.

This scale selection is a synthetic operator-verification fixture, not a
representative calibration method or a final model-accuracy decision.

## Run

Requires the environment and saved float fixture from ../deit-tiny-inspect,
host cc/Clang, riscv64-elf-gcc/binutils and standard Spike.

```sh
cd experiments/deit-linear-int8
sh run.sh
```

MODEL_PYTHON can select a compatible Python interpreter with NumPy/PyTorch.
The script generates reference data/header, checks native C with UBSan, builds
the bare-metal ELF, runs Spike, checks every selected accumulator/output, and
profiles the marked call interval. No network is required after setup.

Expected results:

```text
Prepared full operator reference (197, 768), Spike tile M=2 K=192 N=8
Bound: |every partial accumulator| <= 1,069,197; no INT32 overflow
Requantization tests: 308; first output acc=-6954, INT8=-10
PASS: native C, INT32/INT8 outputs and requantization; UBSan enabled.
PASS: standard Spike, model-derived INT8 Linear tile and requantization.
Measured tile: 3,072 MACs, 22,271 retired instructions, 0.137937 MAC/instruction
```

## Verification and measurement boundaries

- NumPy INT64 full-operator dot/bias reference, plus an independent Python
  arbitrary-precision scalar dot check for all 16 exported outputs.
- C checks all 16 INT32 accumulators and 16 INT8 post-requantization outputs.
- 308 directed/random scalar requantization cases: positive/negative ties,
  INT32 endpoints, zero multiplier, left/right shifts, INT8 saturation; seed6455.
- Native C passed with UndefinedBehaviorSanitizer; the same C passed on Spike.
- Measurement counts only Spike commit records, from kernel_begin inclusive
  to kernel_end exclusive; handles helper calls by runtime interval rather than
  filtering only the linear_int8 function's address range.
- Interval includes argument setup, call/return, function overhead, dot/bias,
  requantization, and both accumulator and INT8 output stores. The accumulator
  store is verification instrumentation and must be held equal in comparisons.
- Correctness loops, the 308 scalar tests, startup/BSS clear, boot ROM, tohost
  and exit spins are excluded. The trace contains them; do not count log lines.
- 3,072 executed mulw instructions agree with the 3,072 mathematical MACs in
  this generated kernel. Compressed instructions count once each.
- No cycles, cache misses, CPI, Fmax, area or CPU speedup are measured.
- These counts are for a small tile and include significant per-call/output
  overhead. Do not extrapolate them to a full model without measurement.

## Files

- prepare.py: quantization, independent integer reference and fixture export.
- kernel.c / kernel.h: ordinary scalar C Linear plus integer requantization.
- main.c: directed tests, measurement markers, accumulator/output comparisons.
- run.sh: full reproducible build/verification/profiling commands.
- profile.py: retired-instruction interval counting.
- build/vectors.h: input, weights, bias, Q31 parameters, expected outputs/tests.
- build/reference.json and operator-reference.npz: quantization metadata and
  full host-reference arrays, explicitly marked as synthetic-fixture verification.
- build/profile.json: actual retired count and mnemonic mix.
- build/linear.elf/.disasm/.map, start.S, link.ld, elf-info.txt, versions and
  source-sha256.txt: reproducibility and address/ISA evidence.

The linker uses the earlier 32 MiB Spike layout with a single RWX segment;
this is not a validated CV-Wally linker configuration.

## Next

Walk through the scalar C loop and verify how one output is computed. Then
establish comparable configurations for custom MAC and requantization in
Spike. This one operator tile does not satisfy full-workload evaluation;
second operators, complete block/model, dataset calibration/accuracy, custom
instruction tests and README/slides remain outstanding.

Primary sources for the selected arithmetic:
- https://github.com/tensorflow/tensorflow/blob/v2.19.0/tensorflow/lite/kernels/internal/common.cc
  MultiplyByQuantizedMultiplier under TFLITE_SINGLE_ROUNDING.
- https://github.com/tensorflow/tensorflow/blob/v2.19.0/tensorflow/lite/kernels/internal/quantization_util.cc
  QuantizeMultiplier (double precision offline path).

These experiment files were generated with AI assistance; follow course policy
and understand/verify their assumptions before submission.
