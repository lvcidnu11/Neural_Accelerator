# Reproduction guide

## Supported setup

Run commands from the repository root unless stated otherwise. The supported
host is macOS arm64 with Python 3.13, Apple Clang, Git, curl, PATH
`riscv64-elf-gcc`/binutils and Spike with matching development headers. Exact
validated versions and model hashes are in README and EXPERIMENTS. Setup does
not install Spike or the standalone PATH RISC-V compiler.

```sh
python3 scripts/project.py setup
python3 scripts/project.py test
python3 scripts/project.py bench
python3 scripts/project.py full
python3 scripts/publish_results.py
```

`setup` creates `experiments/deit-tiny-inspect/.venv`, installs its committed
`requirements-lock.txt`, downloads the pinned TFLM revision and dependencies,
checks archives for GNU Make/xPack GCC, and verifies the official checkpoint's
full SHA256. Downloads and Python packages remain local and ignored.
The checkpoint URL and hash are in `download_weights.py`; setup does not need
a dataset. Dataset accuracy evaluation is a separate unfinished milestone.

The commands were rerun successfully in this course repository on 2026-10-25
using existing local dependency caches. An empty-machine clone has not been
independently tested. Preserve this distinction when reporting reproducibility.
After setup, the measured stages operate on local files without intentionally
downloading new model/runtime dependencies.

## Stage dependencies and pass criteria

| Stage | Prerequisites | Success evidence |
|---|---|---|
| `fixtures` | setup | Strict checkpoint load, finite float outputs, saved NumPy integer Linear reference |
| `test` | setup; regenerates fixtures | ISA fast/logged passes, bounded illegal-instruction controls, tail/alignment cases, 640 blocking/fusion cases on host UBSan and Spike |
| `bench` | setup; regenerates fixtures | All configurations match reference; invoke counters agree with histogram totals; saved FC and optimization comparisons |
| `full` | setup; regenerates fixtures | Builds its own libraries/plugin, then host/UBSan/Spike check 14 nodes and 1,000 exact logits |
| `publish_results.py` | completed test + bench + full | Exports five measured profile groups and provenance into `results/reproduced/` |

The final console line should say `Completed: <stage>`. On failure the entry
point exits nonzero, prints a log path and includes the relevant log tail.
`publish_results.py` refuses to export an incomplete evidence set. Exporting
results does not execute or validate the workloads again.

To run the complete sequence after setup:

```sh
python3 scripts/project.py all
python3 scripts/publish_results.py
```

Running `full` does not require first running `bench`: it builds the host and
RISC-V TFLM libraries through `--library-only` and compiles the custom plugin.
The `test` path retains the earlier qmac-only experiment to cross-check the
standalone scalar/qmac counts before testing combined requantization.

## Output locations

| Location | Contents |
|---|---|
| `build/logs/` | Root stage logs; paths printed by the entry point |
| `experiments/deit-tiny-inspect/results/` | Checkpoint manifest, float fixture, shapes and package freeze |
| `experiments/deit-linear-int8/build/` | Integer operator vectors/reference and standalone artifacts |
| `experiments/qmac8-requant-spike/build/` | Extension, ISA vectors, negative controls, ELFs, traces and comparison |
| `experiments/tflm-linear/build/` | Per-shape/per-configuration graph, ELF, profile, disassembly, optimization tables |
| `experiments/deit-full-int8/build/` | Model/reference, host/UBSan/RISC-V builds, full profile and manifest |
| `results/reproduced/` | Small sanitized JSON profiles and provenance intended for Git |

Generated manifests record exact flags, commands, source/model hashes and
compiler/simulator information. Compact exported profiles are not a replacement
for those full local artifacts when diagnosing a discrepancy. Reruns may
replace stage logs and generated outputs; archive an experiment's build folder
outside the submission if retaining multiple local trials is necessary.

## Expected values

On the validated toolchain, full FC1 counts are 239,507,000 / 32,997,457 /
26,491,730 for scalar / qmac-only / qmac+requant. Shared-kernel B=4 fused
FC1→GELU uses 17,228,262; full inference uses 862,447,442 instructions.
All final integer outputs must match; the exact counters can change with
compiler/simulator versions. Investigate such differences before changing a
reference number. Never disable correctness or ROI consistency assertions to
force a benchmark to pass.

See [results.md](results.md) for the full tables, including small tiles and
instruction mix. The same model/input hash and the same measured region matter
as much as a matching headline number.

## Troubleshooting

| Symptom | Check / resolution |
|---|---|
| Missing compiler or `spike` | Confirm prerequisites on PATH; setup only downloads the separate xPack compiler used by TFLM |
| Missing `riscv/extension.h` or plugin load errors | Use headers from the running Spike build; set `SPIKE_PREFIX` to that installation prefix |
| Illegal instruction in custom workload | Load the matching extension with `--extlib=... --extension=xqmac8`; do not load old/new qmac libraries together. Negative-control tests intentionally trap |
| Python package/import error | Use the local `.venv/bin/python` through the root entry point; recreate setup if Python changed |
| macOS Make rejects newer syntax | Use downloaded GNU Make 4.4.1 through the scripts, not Apple's older `/usr/bin/make` |
| Checkpoint/hash assertion fails | Check the actual downloaded file and pinned checksum; do not relax the assertion |
| Counter/histogram assertion fails | Confirm the validated Spike revision and instruction-budget behavior described in EXPERIMENTS |
| `profile`/`cProfile` import collision | Full-model profiling is named `profile_spike.py`; do not restore a local `profile.py` in that directory |
| Large files appear in Git status | Review `.gitignore`; do not force-add environment, vendor, weights or build directories |

The standalone ISA scripts accept `RISCV_CC`, `RISCV_SIM`, `CXX`,
`SPIKE_PREFIX` and `MODEL_PYTHON`. These are not universal overrides for the
TFLM workflow: its current scripts use the pinned xPack path, local environment
and macOS dynamic-library tooling. Porting that workflow requires code changes.

## Optional standalone RTL reproduction

With Icarus Verilog, Verilator and Yosys installed, from the repository root:

```sh
sh experiments/qmac-design/run.sh
```

This regenerates vectors, simulations, lint, generic netlists and compact
summary under that experiment. It is separate from Spike tests and has not
established CPU timing or SAED physical area.
