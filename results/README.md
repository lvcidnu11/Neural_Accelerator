# Measured Spike evidence

Start with [spike-summary.json](spike-summary.json) and the
[results overview](../docs/spike-results.md): the complete-model progression is
B0 9,115,503,491 → B1 6,095,818,622 → X1 890,189,414 → X2 830,264,926 →
TPU 214,329,448 CPU retired instructions. The summary derives MAC/instruction
from saved measurements, validates model/input hashes and reference checks,
and keeps full-model and FC1-only scopes separate. No RTL data are included.

## Earlier operator studies

`reference-2026-10-25/` contains the previously verified profiles before source
relocation into the course repository. `provenance.json` records original raw
profile hashes and source hashes. Absolute source-root paths in JSON have been
replaced with `<repo>`; no ELF, weights, model, toolchain or commit trace is
included. Read [EXPERIMENTS.md](../EXPERIMENTS.md) for measurement boundaries.

After rerunning the relocated code, `reproduced/` contains the corresponding
new profiles. This distinguishes preserved measurements from a new run.
Use `python3 scripts/publish_results.py` to refresh that directory after all
stages pass. Counts may change with compiler/simulator revisions; investigate
any change rather than assuming the old counts still apply.

| File | Executed scope |
|---|---|
| `isa-tile.json` | Standalone 2×192×8 Linear kernel, three configurations |
| `fc1.json` | TFLM tile and full 197×192×768 FC1, three configurations |
| `fc1-gelu.json` | Same configurations with GELU LUT in every row |
| `blocking-fusion.json` | Two shapes × three blocking factors × fusion on/off |
| `full-inference.json` | Complete optimized DeiT-Tiny, one synthetic input |

These earlier operator profiles do not demonstrate hardware cycle speedup,
physical area or calibrated accuracy. The separately measured scalar-versus-
custom full-model instruction-count comparison is linked above. Full inference includes
stage-hash checking. Register blocking B=1 is a compile-time specialization,
not the assignment's software-optimized scalar B1 baseline.
