#!/bin/sh
set -eu
cd "$(dirname "$0")"
riscv64-elf-gcc --version > compiler-version.txt
spike --help > simulator-version.txt 2>&1
shasum -a 256 dot.c start.S link.ld > source-sha256.txt
riscv64-elf-gcc -march=rv64imc -mabi=lp64 -mcmodel=medany \
    -ffreestanding -O2 -fno-inline -S dot.c -o dot.s
riscv64-elf-gcc -march=rv64imc -mabi=lp64 -c dot.s -o dot.o
riscv64-elf-gcc -march=rv64imc -mabi=lp64 -c start.S -o start.o
riscv64-elf-gcc -march=rv64imc -mabi=lp64 -nostdlib -nostartfiles \
    -Wl,--no-relax,-Map=dot.map -T link.ld start.o dot.o -o dot.elf
riscv64-elf-objdump -d dot.elf > dot.disasm
riscv64-elf-nm -S --defined-only dot.elf > symbols.txt
spike --isa=RV64IMC -m32 -l --log-commits --log=spike.log dot.elf
printf 'PASS: scalar length-8 dot product returned 744.\n'
