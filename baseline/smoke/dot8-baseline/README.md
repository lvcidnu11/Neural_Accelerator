# Length-8 scalar baseline

Measured 2026-10-23. Run `sh dot8-baseline/run.sh` from the parent smoke folder.
Independent copies preserve the original four-element experiment.

A = {1,3,5,7,9,11,13,15}; W = {2,4,6,8,10,12,14,16}; n=8.
Host arithmetic: sum(A[i]*W[i]) = 744. Startup checks the returned value
against 744; Spike exited successfully. This remains a signed INT8 scalar
test using standard RV64IMC instructions; no custom qmac is implemented.

Flags: O2, fno-inline, rv64imc, lp64, medany, freestanding; no standard
startup/libraries; linker relaxation disabled. The run script records versions,
source hashes, disassembly, symbols and commit trace. Kernel-only commit count
within dot8_t's ELF symbol interval was 61, matching 4 setup + 8*7 loop + 1
return. Caller/startup/tohost/hang excluded. This is not a cycle measurement.

Next correctness cases should include negative values and tails, not only
positive aligned groups of eight. The assignment requires INT8 W8A8 and
directed/randomized custom-instruction verification. Midterm grading names
ISA and Spike with custom instructions; full-project requirements include
both packed MAC and requantization. Detailed checkpoint scope must follow any
additional course handout, if provided.
