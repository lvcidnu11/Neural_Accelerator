# Assignment mapping and outstanding work

The authoritative assignment is `rv64imc_quantized_nn_term_project.md`.
Status below separates working software from final course completion.

| Requirement | Evidence/location | Status |
|---|---|---|
| Bare-metal C/C++, RV64IMC, static memory | `tflm-linear/start.S`, `link.ld`, `deit-full-int8/*.cc` | Implemented; Spike verified |
| Quantized model/conversion scripts | `deit-full-int8/export_model.py`, `deit-linear-int8/prepare.py` | Implemented; synthetic scales only |
| Spike custom packed MAC and requantization | `qmac8-requant-spike/extension.cc`, `ISA.md` | Implemented and verified |
| Short instruction verification tests | generators, `instruction_tests.c`, `requant_tests.c` | Directed/random, overlap/x0/wrap, traps |
| Reproducible benchmark scripts | `scripts/project.py`, experiment build/run/profile scripts | Implemented; macOS arm64 supported |
| B0 / X1 / X2 incremental comparison | full FC1 and FC1→GELU, `results/` | Measured |
| Software-optimized scalar B1 | complete representative scalar optimization study | Still needed; custom blocking B=1 is not assignment B1 |
| Three largest bottlenecks | instruction mixes available; operator attribution across full model | Analysis/structured report still needed |
| Two major operators and full workload coverage | FC1 ablations, complete model correctness/profile | Need structured comparative second-operator results |
| Instruction mix, memory instructions, MAC/instruction, utilization, code size | saved profile JSON, sizes, disassembly | Software metrics available; operator contribution incomplete |
| Instruction-reduction threshold on major FC | FC1 scalar 239,507,000 vs custom 26,491,730 | Exceeds 2x instruction-count ratio; no cycle claim |
| Lane design exploration | existing `experiments/qmac-design` standalone RTL | Partial; no CPU integration |
| CPU cycle/CPI, representative kernel, Fmax/critical path/area, efficiency | CVW integration and SAED flow | Not completed |
| Dataset calibration/accuracy | representative images and validation evaluation | Not completed; float/quantized top-1 differ on current fixture |
| Full-model baseline and end-to-end speedup | fair scalar versus custom full-model runs | Not completed |
| GitHub run instructions and expected metrics | root README, DESIGN, EXPERIMENTS, results | Prepared for review |
| 5–6 minute slides; final report | presentation and report artifacts | Not completed |
| Assistance disclosure / course policy | AI_USAGE.md | Draft disclosure; student must check course rules |

Suggested next milestones:

1. Submission checkpoint: verify the relocated source, review Git diff, push
   the code/docs/evidence and present the implemented scope honestly.
2. Complete the software study: fair scalar B1, second operator ablation,
   full-model operator attribution and, when a dataset is available, calibration
   and accuracy. Completion means reproducible comparisons and explicit errors.
3. Complete the hardware study: integrate chosen ISA datapaths into CVW,
   validate representative kernels, measure cycles and synthesize using the
   specified technology. Completion means correctness plus timing/area evidence.
4. Write final report/slides tied to those measurements and explain the design.
