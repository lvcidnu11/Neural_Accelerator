> Repository entry point and current status: [root README](../../README.md).
> Experiment-specific results below retain their stated measurement scope.

# DeiT-Tiny: official float checkpoint and operator inspection

Verified 2026-10-23. This is the first model milestone, not a W8A8 model,
bare-metal inference, Spike model execution, or an accuracy benchmark.

## Location and environment

This experiment is part of the course repository. Dependencies, weights and
generated fixtures stay local and are ignored by Git. From the repository root:

```sh
cd experiments/deit-tiny-inspect
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_weights.py
sh run.sh
```

Once installed/downloaded, run.sh is offline. On the original machine the
environment and weights are already ready; rerun just `sh run.sh`. The complete
resolved dependency set is saved in results/requirements-lock.txt; a clean
reproduction can install that lock instead of resolving transitive versions
again. The tested interpreter is Python 3.13, PyTorch 2.9.1, torchvision 0.24.1,
timm 1.0.30, CPU backend, one thread, seed 6455.

## Model and provenance

- Exact variant: deit_tiny_patch16_224, non-distilled, 1,000 classes.
- Official code: https://github.com/facebookresearch/deit/blob/main/models.py
- Official checkpoint:
  https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth
- Verified checkpoint SHA256:
  a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97
- Downloaded official source is saved as evidence, never imported or executed.
- Local timm instantiates the architecture. Official state dict loads strictly,
  with all expected tensors matching. Source content hash and checkpoint hash
  are recorded in results/model-manifest.json.

## Verified structure

One synthetic normal-random float input of shape [1,3,224,224] produced finite
logits of shape [1,1000]. This tests loading and numerical execution, not
ImageNet accuracy or calibration quality. No real image preprocessing has
been validated yet. The timm preprocessing metadata is recorded as a candidate
configuration; confirm it against the official checkpoint's evaluation pipeline
before using a dataset.

| Property | Observed |
|---|---:|
| Parameters | 5,717,416 |
| Float parameter bytes | 22,869,664 |
| Transformer blocks | 12 |
| Image patches | 14 x 14 = 196 |
| Tokens including class token | 197 |
| Embedding width | 192 |
| Attention heads | 3 |
| MLP hidden width | 768 |

For a linear layer, X[M,K] * W[N,K]^T + bias[N] = Y[M,N]. Each output is
one K-element dot product. Actual first-block forward shapes:

| Operator | M | K | N | Shape-derived MACs |
|---|---:|---:|---:|---:|
| blocks.0.attn.qkv | 197 | 192 | 576 | 21,786,624 |
| blocks.0.attn.proj | 197 | 192 | 192 | 7,262,208 |
| blocks.0.mlp.fc1 | 197 | 192 | 768 | 29,048,832 |
| blocks.0.mlp.fc2 | 197 | 768 | 192 | 29,048,832 |

K=192 and K=768 are divisible by eight: a packed-dot path could process 24 or
96 groups per output, respectively. Those are grouping counts, not measured
custom-instruction counts or cycle speedups. Other operations, including the
197-token attention reductions, can still need remainder handling.

The counted Conv/Linear/attention matmul workload is 1,253,683,200 MACs for one
forward at batch 1. One multiply and its accumulation count as one MAC.
Conv/Linear counts come from runtime tensor shapes; functional attention
matmul counts are derived from observed token count and head configuration.
This excludes bias, LayerNorm, GELU, Softmax, residual additions, and other
non-MAC work. It is NOT dynamic instruction profiling or a runtime bottleneck
measurement. Do not label MAC totals as FLOPs without defining the convention.

## Correctness checks and artifacts

- Strict loading of the official pretrained state dict.
- Expected input/output shapes, finite logits, block/embedding dimensions.
- Four first-block Linear captures checked against X @ W.T + bias with
  floating-point tolerance (rtol=1e-4, atol=1e-5), not integer bit-exactness.
- results/inspection.json: structure, version info, observed operator shapes
  and shape-derived work counts.
- results/linear-shapes.csv: all Linear M/K/N shapes for the 12 blocks and head.
- results/float-fixtures.pt: synthetic input, logits, four first-block Linear
  inputs, weights, biases and outputs, explicitly marked unquantized.
- results/model-manifest.json, source-sha256.json, requirements-lock.txt:
  source/weight provenance and reproducibility.

Next: establish the model quantization/calibration and independent integer
reference. Start porting a representative model Linear operator to bare-metal
C, validate it in standard Spike, then use it to evaluate custom instructions.
These float fixtures alone do not establish meaningful quantized-model accuracy,
full-block integer execution, or completion of the quantized-model deliverable.
