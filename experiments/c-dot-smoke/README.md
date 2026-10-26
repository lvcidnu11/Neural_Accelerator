> Repository entry point and current status: [root README](../../README.md).
> Experiment-specific results below retain their stated measurement scope.

# C-generated RV64IMC assembly smoke test

Verified on 2026-10-05 with GCC 16.2.0 and Spike 1.1.1-dev.

## Run

From this directory:

```sh
sh run.sh
```

Expected: `PASS: main returned the expected dot product (100).`
The script stops on a build error or simulator failure. A hung simulator can be
interrupted with Ctrl+C. Spike normally exits after receiving the tohost signal.

## What executes

```text
dot.c -> dot_c.s -> dot_c.o
start_c.S -> start_c.o
start_c.o + dot_c.o + link_c.ld -> dot_c.elf -> Spike
```

`dot.c` was copied unchanged from
`baseline/smoke/dot.c`.
`dot_c.original.s` preserves the previously generated assembly from that directory.
All builds here are separate from those original files.

The input is A={1,3,5,7}, W={2,4,6,8}, n=4.
Expected result: 1*2 + 3*4 + 5*6 + 7*8 = 100.
The earlier handwritten dot.S used -1 as its first A element and checked 96;
that expected value must not be reused for this C program.

## Necessary changes

- Compiler name: this Mac provides `riscv64-elf-gcc`.
- The old assembly assembled successfully but linking at 0x80000000 failed with
  `relocation truncated to fit: R_RISCV_HI20 against W`. Regenerating with
  `-mcmodel=medany` gives PC-relative array addresses suitable for this layout.
  Passing the flag only at link time cannot change existing assembly instructions.
- `-ffreestanding` selects the bare-metal compilation environment. Without it,
  this installed compiler's stdint.h attempted to include a missing libc header.
- `-O0` keeps the original debugging optimization level.
- New `_start` initializes a 16-byte-aligned stack at 0x82000000, clears .bss,
  calls main, compares a0 with 100, and writes 1 for pass or 3 for failure to tohost.
- The linker explicitly places constants, data, .bss, and host communication symbols.
  Spike loads initialized sections directly from the ELF; no ROM-to-RAM .data
  copy is needed for this experiment.
- `--no-relax` keeps address sequences explicit and avoids gp-relative relaxation;
  this small wrapper does not initialize gp. Revisit startup for larger programs.

GCC reference: https://gcc.gnu.org/onlinedocs/gcc/RISC-V-Options.html
Spike reference: https://github.com/riscv-software-src/riscv-isa-sim

## Evidence

The successful `spike.log` contains four executed `mulw` instructions,
`x10 0x0000000000000064` on return (a0=100), and a store of 1 to tohost.
`dot_c.disasm` shows the linked instructions; `dot_c.map` shows memory placement.
`compiler-version.txt` and `simulator-version.txt` record the installed tools.

The linker emits a warning about one writable/executable RAM load segment; this
simple bare-metal linker layout intentionally puts the program in a single RAM
segment. This run validates functionality in Spike only. No CV-Wally cycle count
or performance improvement was measured. The simulator may log spins at `hang`
while waiting for host polling; do not count the entire log as kernel instructions.

## Next milestone

Inspect the trace within dot8_t, then build an optimized scalar baseline and
establish explicit profiling boundaries. This four-element debugging example is
not a complete benchmark or a custom MAC implementation.
