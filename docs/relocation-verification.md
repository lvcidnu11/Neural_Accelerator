# Relocation verification — 2026-10-25

The course repository is now the working copy. Sources and local dependency
caches were copied independently; no symlink to the ChatGPT mirror is needed.
The local virtual environment entry points were relocated as well.

Passed from this repository: `scripts/project.py setup`, `test`, `bench`, and
`full`; exported profiles with `scripts/publish_results.py`. All staged
experiment instruction counts reproduce the previous measurements. Full-model
instruction mix and model SHA256 also match, with 862,447,442 instructions,
14 checked nodes, 12 complete blocks and 1,000 exact output logits.

The relocation exposed missing startup-file prerequisite checks and a Python
`profile.py`/standard-library collision. Those are fixed. Full-model startup
now builds its own TFLM libraries and plugin instead of depending on artifacts
from a prior operator benchmark.

See `results/reproduced/verification.json` and measured profiles for evidence.
Local caches were reused; this is not a claim that a separate empty-machine
clone was tested. Generated full logs, build manifests and ELFs remain ignored
under experiment `build/` directories. No Git staging, commit or push was done
during that verification; subsequent local commits are documented in
`docs/submission.md`.

## Commit/export check — 2026-10-26

The committed source inventory was audited, then `git archive HEAD` was
extracted to a temporary clean directory without local dependency caches.
`experiments/c-dot-smoke/run.sh` built and passed on stock Spike (expected
return value 100). The startup `.S` file is explicitly retained despite the
case-insensitive `.s` output ignore rule. This checks the exported baseline
source; it is not a new empty-machine full-model reproduction. Documentation
links/anchors, quoted measured counts, JSON/counter totals and commit contents
were also checked. The local history is grouped into nine milestone commits;
no remote push is part of this preparation.
