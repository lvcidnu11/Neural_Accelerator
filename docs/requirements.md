# Requirements and project status

For the seven-item ISA checkpoint supplied on 2026-11-03, use the explicit
mapping and run instructions in [checkpoint.md](checkpoint.md). The checkpoint
requires source, quantization/conversion, Spike ISA implementation, short
instruction tests, reproducible full-workload evaluation, a 5–6 minute deck,
and a GitHub README with expected counts. The provided checkpoint instructions
explicitly do **not** require CV-Wally or PPA evaluation.

| Checkpoint item | Current evidence | Status |
|---|---|---|
| Bare-metal C/C++ workload | `experiments/deit-full-int8/`, `experiments/tflm-linear/` | Complete 12-block integer inference, Spike verified |
| Quantized model or Python conversion | `experiments/deit-full-int8/export_model.py` | Implemented; synthetic fixture and project scales |
| Spike/custom ISA source | `experiments/qmac8-requant-spike/`, `experiments/tpu-spike/` | qmac8/requant8 verified; TPU functional exploration |
| Short instruction tests | `experiments/qmac8-requant-spike/run.py` | Directed/randomized and invalid-encoding controls |
| Full-workload evaluation | `scripts/project.py checkpoint`, `results/incremental-inference-2026-11-03/` | B0/B1/X1/X2/TPU measured; one synthetic input |
| 5–6 minute presentation | Student-number presentation in the local project folder; not tracked | English slides and speaker notes |
| README and expected measurements | `README.md`, `docs/checkpoint.md`, `docs/reproduction.md` | Commands, provenance and saved counts documented |

The full rerun is validated on macOS arm64 with the listed dependencies; a clean
machine clone or a complete rerun of the new single-command entry point has
not been independently verified. Current results match 14
stage hashes and 1,000 integer logits for one input. They do not establish
calibrated dataset accuracy. Spike retired counts are ISA measurements, not
hardware speedup.

The broader course project may later include CV-Wally integration, physical
implementation and a final report. Those remain separate from this checkpoint;
consult the instructor's current final-stage requirements rather than treating
these items as checkpoint prerequisites. AI assistance is recorded in
[AI_USAGE.md](../AI_USAGE.md).
