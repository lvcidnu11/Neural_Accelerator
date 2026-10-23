# O2 inlining and RAM comparison

Measured 2026-10-23 using GCC 16.2.0. Reproduce from the smoke directory:
`sh inline-comparison/run.sh`. The script snapshots dot.c, start.S and link.ld
into inputs and records their SHA256 hashes. Only -fno-inline differs between
the two compiler configurations; startup, ISA, ABI, linker script, disabled
linker relaxation, and Spike configuration are identical. Both runs check
main's return value against 100 and passed in Spike.

| Bytes | O2 inline allowed | O2 -fno-inline |
|---|---:|---:|
| main function | 50 | 26 |
| standalone dot8_t function | 38 | 38 |
| .text section (including startup) | 132 | 108 |
| .srodata section (includes internal alignment) | 12 | 12 |
| .tohost section (includes internal alignment) | 72 | 72 |
| Sum of allocated section sizes | 216 | 192 |
| Single LOAD segment p_memsz (includes gaps between sections) | 264 | 200 |
| Compiler-reported stack frame, main | 0 | 0 |
| Compiler-reported stack frame, dot8_t | 0 | 0 |

Inlining increased code by 24 bytes in this program. The occupied static RAM
span increased by 64 bytes because .tohost is aligned to 64 bytes: its start
moved from 0x80000080 to 0x800000c0. The LOAD segment ends are 0x800000c8
and 0x80000108 respectively. No runtime stack usage was measured; the .su files
report compiler frame sizes, and inspection shows neither optimized path nor
startup allocates a stack frame. The configured RAM remains 32 MiB in both
runs. Defining _stack_top does not allocate all of that memory to this program.

The GNU size summary's text column is .text + read-only .srodata here (144 vs
120), rather than the .text section alone. Its dec column is not a measurement
of total runtime peak RAM. Debug metadata, symbol tables and ELF file padding
are not automatically target RAM consumption. Use sections.txt and layout.txt
for section/segment memory layout; do not compare filesystem file sizes.

This result applies to the inspected source/compiler/linker setup. It is not
a general conclusion that inlining always increases RAM by 64 bytes. No cycle
speedup was measured. Keep function-level profiling boundaries explicit:
inline/main contains its own loop, whereas noinline/main tail-jumps to dot8_t.

## Dynamic instruction comparison

Run `python3 inline-comparison/profile.py` from the smoke directory after the
build, or rerun run.sh. Spike versions inherit the installed simulator used
by the smoke experiment. profile.py uses the generated ELF symbol ranges and
counts commit records only. Boot ROM instructions are excluded. Counting ends
at the commit of the store of 1 to tohost, so host-polling hang loops are excluded.
Both variants completed successfully.

| Retired instructions | Inline | No inline |
|---|---:|---:|
| main itself | 36 | 7 |
| standalone dot8_t executed | 0 | 33 |
| main plus its executed callee | 36 | 40 |
| startup and result reporting | 11 | 11 |
| _start through success store | 47 | 51 |

Both execute 4 MACs and 8 byte loads in the compute region. The 4-iteration
loop itself is 28 instructions in both cases. Inline main is 7 setup + 28
loop + 1 return. No-inline main is 7 setup/tail-jump instructions; dot8_t is
4 setup + 28 loop + 1 return. Each trace total was checked against these manual
counts. Inline therefore saves 4 retired instructions (10% of the 40-instruction
non-inline compute path), while its code section is 24 bytes larger. This is
not a cycle speedup measurement. Inlining's advantage here includes using the
known positive n=4 to eliminate the general function's n<=0 guard; it is not
only call/return elimination. The tail call also means no separate main return
executes in the no-inline path.

Scope: a single 4-element teaching input, not representative model profiling
or a general-purpose trace parser. Assembly forms and encoding checks in
profile.py are specific to these generated optimized programs. Runtime logs
are in each variant's spike.log; aggregate values are in profile-results.json.
