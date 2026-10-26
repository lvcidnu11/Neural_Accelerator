# Commit organization and pushing

The submission is organized into reviewable milestones using the current
verified source snapshots. Existing baseline commits remain unchanged. The
new commits are created now with their real timestamps; they do not claim to
reconstruct historical intermediate source states that were not preserved.

| Order | Commit scope | Main files |
|---|---|---|
| 1 | Bare-metal C checkpoint, assignment and generated-file exclusions | `.gitignore`, assignment, `experiments/c-dot-smoke/` |
| 2 | Official model inspection and scalar INT8 reference | `deit-tiny-inspect/`, `deit-linear-int8/` |
| 3 | Packed qmac8 Spike extension and tests | `qmac8-spike/` |
| 4 | Custom requant8 semantics, tests and incremental comparison | `qmac8-requant-spike/` |
| 5 | Standalone 1/2/8-lane RTL design exploration | `qmac-design/` |
| 6 | TFLM FC/GELU integration, blocking/fusion and verification | `tflm-linear/` |
| 7 | Complete project DeiT-Tiny graph/reference and inference | `deit-full-int8/` |
| 8 | Unified reproduction entry points and measured evidence | `scripts/`, `results/` |
| 9 | Project overview, design rationale, protocol and completion status | README, DESIGN, EXPERIMENTS, AI_USAGE, `docs/` |

The TFLM stage contains the integrated shared kernel, including blocking and
fusion; its measured ablation separates those effects. It is not presented as
several artificial feature commits containing invented earlier versions.
Some early commits refer forward to the final documentation/root entry point;
individual experiment scripts remain the source of their build/test workflows.

Downloaded dependencies, weights, Python environments, generated ELFs/traces,
RTL netlists and simulator binaries are ignored. Compact results, provenance
and RTL summary/version information are included. Original tracked artifacts
under `baseline/smoke` remain as they were; ignore rules do not remove history.

## Review the local history

```sh
git status --short
git log --oneline --reverse 7387ee2..HEAD
git diff --stat 7387ee2..HEAD
git diff --check 7387ee2..HEAD
```

Inspect a specific commit with `git show <hash> --stat` or `git show <hash>`.
The code/docs commit sequence does not imply that the unfinished hardware or
accuracy requirements have been met; see [requirements.md](requirements.md).
AI assistance is recorded in [AI_USAGE.md](../AI_USAGE.md).

## Push after review

```sh
git branch --show-current
git remote -v
git push origin main
```

The prepared branch is `main`, remote is the existing
`ScGong04/Neural_Accelerator` repository. If your branch/remote differs, use the
actual values. Creating the local commits does not itself push anything.
Do not squash the milestones if you want to preserve this review sequence.
If pushing is rejected because the remote advanced, fetch and inspect the
branch difference before merging/rebasing; do not force-push to bypass it.
