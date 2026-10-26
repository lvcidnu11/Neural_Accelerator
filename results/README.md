# Measured software evidence

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

No profile demonstrates hardware cycle speedup, physical area, calibrated
accuracy or scalar-versus-custom full-model speedup. Full inference includes
stage-hash checking. Register blocking B=1 is a compile-time specialization,
not the assignment's software-optimized scalar B1 baseline.
