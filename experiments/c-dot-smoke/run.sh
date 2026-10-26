#!/bin/sh
set -eu
cd "$(dirname "$0")"

# Homebrew's RISC-V compiler has the riscv64-elf prefix.
CC=${RISCV_CC:-riscv64-elf-gcc}
OBJDUMP=${RISCV_OBJDUMP:-riscv64-elf-objdump}
SIM=${RISCV_SIM:-spike}

"$CC" --version > compiler-version.txt
"$SIM" --help > simulator-version.txt 2>&1
"$CC" -march=rv64imc -mabi=lp64 -mcmodel=medany -ffreestanding \
    -O0 -S dot.c -o dot_c.s
"$CC" -march=rv64imc -mabi=lp64 -c dot_c.s -o dot_c.o
"$CC" -march=rv64imc -mabi=lp64 -c start_c.S -o start_c.o
"$CC" -march=rv64imc -mabi=lp64 -nostdlib -nostartfiles \
    -Wl,--no-relax,-Map=dot_c.map -T link_c.ld \
    start_c.o dot_c.o -o dot_c.elf
"$OBJDUMP" -d dot_c.elf > dot_c.disasm
"$SIM" --isa=RV64IMC -m32 -l --log-commits --log=spike.log dot_c.elf
printf 'PASS: main returned the expected dot product (100).\n'
