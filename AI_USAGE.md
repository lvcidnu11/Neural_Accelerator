# AI assistance disclosure

Codex/ChatGPT assisted with conceptual explanations, implementation and review
of the quantized kernels/custom Spike instructions, test/reference generation,
benchmark automation, full-model integration and repository documentation.
The student directed the design discussion and is responsible for reviewing,
understanding and defending the submitted work.

Generated explanations and code are not independent experimental evidence.
Claims in this repository must be supported by executable tests, saved measured
profiles, source code or the assignment. No measurement should be represented
as hardware performance when it is only functional Spike instruction counting.

This disclosure records assistance; it does not establish that the course
permits every use. Check the instructor's academic-integrity and attribution
requirements before submission and amend this document with any required
session, model or contribution details. Do not claim sole unaided authorship.

The project uses official DeiT pretrained weights offline and pinned upstream
TFLM/dependencies downloaded during setup. Their source and license files remain
with those downloads; they are not relicensed by this repository. No weights,
third-party source tree, toolchain or dataset is included in the intended commit.

2026-10-26 TPU branch: Codex assisted with the 8×8 tile descriptor ISA, Spike
functional engine, shared-kernel integration, scalar-oracle/negative tests,
controlled full-model comparison and the separately labeled timing model.
No TPU RTL or synthesized physical measurements were generated.

2026-11-02: Codex assisted with the inline/parameter-reuse ablation, controlled
full-model evaluation, optional kernel integration and an English presentation.
Hardware speedup, area, power and model accuracy remain unmeasured.

2026-11-02 follow-up: Codex assisted with inline code/stack audits, GCC stack
diagnostics, functional instruction-cache sensitivity and memory-tradeoff slides.
The measurements do not establish RTL speedup or a worst-case stack bound.

2026-11-03: Codex assisted with a direct scalar B=4 full-model baseline,
software requantization, independent scalar kernel checks, host UBSan and
stock-Spike measurement, and a six-slide English presentation for five minutes.
The student should distinguish complete-model instruction reductions from
hardware speedup and explain which software optimizations the control retains.

2026-11-03 CV-Wally readiness audit: Codex inspected the local Wally decode,
LSU, EBU and configuration sources, generated a reproducible harness for
the unchanged controller RTL, and documented six decoder checks. It
qualified TPU/software claims and the five-minute English slides/notes.
No upstream Wally source was modified and no full-core accelerator run,
RTL cycle result or area result was produced.

2026-11-03 presentation template update: Codex adapted the user-supplied
black-and-white PowerPoint template, preserving its typography and repeated
elements while replacing stock content with the six-slide project talk.
It retained editable evidence and English speaker notes, and distinguished
measured Spike instruction counts from proposed Wally hardware choices.

2026-11-03 incremental comparison: Codex implemented isolated B0 and X1
source overlays, ran full-inference reference and stock-Spike checks, verified
X1 kernel cases and assembled the five-stage comparison and presentation.
B0 retains a separate GELU LUT. The results do not establish RTL speedup.

2026-11-03 ISA checkpoint packaging: Codex organized the seven-deliverable
checklist, updated reproduction commands and paths, and prepared a separate
checkpoint deck from the student's edited presentation. It corrected the chart
unit, updated English notes and limited the checkpoint scope to ISA/Spike.

2026-11-06 Spike results publication: Codex rebuilt the result overview from
committed measurements, added derived useful MAC/CPU-retired-instruction metrics
and source hashes, and made B0=9,115,503,491 through TPU=214,329,448 the primary
comparison. It distinguished software-optimized B1, historical inline/hoist
variants and FC1-only experiments. This update does not rerun the full model
or publish CV-Wally/RTL experiments.
