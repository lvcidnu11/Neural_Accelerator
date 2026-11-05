# ISA checkpoint submission

This checklist follows the seven checkpoint deliverables supplied by the course
on 2026-11-03. This checkpoint evaluates ISA behavior in Spike; CV-Wally RTL,
PPA, cycles and Fmax are not required for this checkpoint.

| Deliverable | Where to find it | Evidence |
|---|---|---|
| Bare-metal C/C++ workload | `experiments/deit-full-int8/`, `experiments/tflm-linear/` | Full 12-block integer model on RV64IMC in Spike |
| Quantized model or conversion | `experiments/deit-full-int8/export_model.py`, `experiments/deit-tiny-inspect/`, `experiments/deit-linear-int8/` | Pinned pretrained checkpoint, generated integer model and Python reference |
| Spike/custom ISA source | `experiments/qmac8-requant-spike/` and optional `experiments/tpu-spike/` | qmac8 and requant8 semantics, encoding, extension and wrappers |
| Short instruction tests | `experiments/qmac8-requant-spike/run.py`, ISA test generators and C tests | Directed/randomized arithmetic, overlap, x0 and illegal-encoding checks |
| Full-workload benchmark scripts | `scripts/project.py`, `experiments/b0-full-inference-2026-11-03/`, `experiments/x1-qmac-inference-2026-11-03/` | Reproducible B0/B1/X1/X2/TPU runs and correctness checks |
| 5–6 minute presentation | Local project folder; student-number presentation (not tracked) | Six English slides with English speaker notes |
| GitHub instructions and expected values | `README.md`, `docs/reproduction.md`, `results/incremental-inference-2026-11-03/` | Setup, test, measurement and saved results |

## Reproduce

The validated host is macOS arm64, Python 3.13, Apple Clang and the versions
listed in [reproduction.md](reproduction.md). Spike and its matching extension
headers and the PATH `riscv64-elf-*` tools must be installed separately.
The workflow downloads pinned dependencies and the official checkpoint during
setup; downloaded weights, tools, generated models and logs are not committed.
The comparison uses a checked-in frozen kernel from commit `a2bee01`, with a
checksum check. An empty-machine clone has not been independently tested.
The new `checkpoint` entry point was checked for stage order and its constituent
experiments were verified separately with existing dependencies; the combined
command has not yet been rerun end-to-end after packaging.

```sh
python3 scripts/project.py setup
python3 scripts/project.py checkpoint
```

`checkpoint` runs the ISA tests and complete model, builds both Spike plugins,
then measures B0, B1, X1, X2 and TPU with the same model and input. B0 alone
executes the full scalar workload twice and can take many minutes. The run uses
fresh build results, validates model/input hashes and integer reference checks,
and writes `experiments/b0-full-inference-2026-11-03/build/REPORT.md`. The
complete logs and generated ELFs are under the relevant ignored `build/` folders.

To inspect the committed results without running the model:

```sh
python3 experiments/b0-full-inference-2026-11-03/collect.py --saved
```

This writes a summary into that experiment's ignored `build/` folder. Exact
counts can change with compiler or simulator versions; investigate differences
without disabling correctness checks.

## Expected full-inference results

B0 is the 9.116-billion unblocked scalar baseline; B1 is its software-optimized
6.096-billion stage. For useful MAC/instruction and the supporting ablations,
see the [Spike results overview](spike-results.md).

| Configuration | CPU retired instructions |
|---|---:|
| B0: scalar, B=1, separate GELU LUT | 9,115,503,491 |
| B1: software-optimized scalar | 6,095,818,622 |
| X1: B1 + qmac8 | 890,189,414 |
| X2: X1 + requant8 | 830,264,926 |
| TPU functional tile path | 214,329,448 |

Each interval includes 14 stage-hash checks. The five variants use the same
model, quantization and synthetic input and match all 1,000 integer logits. B0
passes stock Spike; X1 passes 640 kernel cases on host UBSan and Spike. B0 uses
a separate noinline GELU LUT function inside the existing custom block node,
not a separate TFLM operator dispatch. B0 still uses a LUT to preserve the
same tested quantized GELU; its comparison with B1 combines blocking, fusion
and inline rather than isolating each software change.

The TPU path is a functional Spike extension: one retired tile command hides
its internal arithmetic and memory operations. These counts are not hardware
cycles or hardware speedups. One synthetic input and project quantization do
not establish dataset accuracy. The later CV-Wally work is outside this ISA
checkpoint.
