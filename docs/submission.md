# Submission and repository organization

The seven-item ISA checkpoint checklist is in [checkpoint.md](checkpoint.md).
The `main` branch includes the functional TPU exploration as well as the
B0/B1/X1/X2 comparisons. See the root
README for the code map and run commands. The presentation is maintained
locally in the project folder and is not part of the Git submission.

Downloaded dependencies, checkpoints, Python environments, generated models,
ELFs and large traces are ignored; pinned setup scripts recreate them. Compact
results and provenance are tracked. The saved results are evidence from the
specified toolchain, not a substitute for running the verification scripts.

To review the current branch:

```sh
git status --short
git diff --check
git log -5 --oneline
```

AI assistance is documented in
[AI_USAGE.md](../AI_USAGE.md).
