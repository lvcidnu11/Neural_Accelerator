#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p inputs
cp ../dot.c ../start.S ../link.ld inputs/
riscv64-elf-gcc --version > compiler-version.txt
shasum -a 256 inputs/* > source-sha256.txt

for variant in inline noinline; do
    mkdir -p "$variant"
    extra=
    if [ "$variant" = noinline ]; then extra=-fno-inline; fi
    riscv64-elf-gcc -march=rv64imc -mabi=lp64 -mcmodel=medany \
        -ffreestanding -O2 $extra -fstack-usage \
        -S inputs/dot.c -o "$variant/dot.s"
    riscv64-elf-gcc -march=rv64imc -mabi=lp64 \
        -c "$variant/dot.s" -o "$variant/dot.o"
    riscv64-elf-gcc -march=rv64imc -mabi=lp64 \
        -c inputs/start.S -o "$variant/start.o"
    riscv64-elf-gcc -march=rv64imc -mabi=lp64 -nostdlib -nostartfiles \
        -Wl,--no-relax,-Map="$variant/dot.map" -T inputs/link.ld \
        "$variant/start.o" "$variant/dot.o" -o "$variant/dot.elf" \
        > "$variant/link.log" 2>&1
    riscv64-elf-size -A "$variant/dot.elf" > "$variant/sections.txt"
    riscv64-elf-readelf -SW -lW "$variant/dot.elf" > "$variant/layout.txt"
    riscv64-elf-nm -S --size-sort "$variant/dot.elf" > "$variant/symbols.txt"
    riscv64-elf-objdump -d "$variant/dot.elf" > "$variant/dot.disasm"
    spike --isa=RV64IMC -m32 "$variant/dot.elf"
    printf '%s: PASS\n' "$variant"
done
riscv64-elf-size inline/dot.elf noinline/dot.elf
python3 profile.py
