#!/bin/sh
set -eu
cd "$(dirname "$0")"
PYTHON=${MODEL_PYTHON:-../deit-tiny-inspect/.venv/bin/python}
mkdir -p build
"$PYTHON" prepare.py
cc -std=c11 -O1 -g -Wall -Wextra -Werror -fsanitize=undefined \
    -fno-sanitize-recover=all -Ibuild kernel.c main.c -o build/native-check
build/native-check
printf 'PASS: native C, INT32/INT8 outputs and requantization; UBSan enabled.\n'
riscv64-elf-gcc --version > build/compiler-version.txt
spike --help > build/simulator-version.txt 2>&1
cp ../c-dot-smoke/link_c.ld build/link.ld
sed -e 's/li t2, 100/li t2, 0/' \
    -e 's/# main returns the dot product in a0; check before reporting success./# main returns zero only when all reference comparisons pass./' \
    ../c-dot-smoke/start_c.S > build/start.S
for source in kernel main; do
    riscv64-elf-gcc -march=rv64imc -mabi=lp64 -mcmodel=medany \
        -ffreestanding -O2 -fno-inline -Wall -Wextra -Werror -Ibuild \
        -S "$source.c" -o "build/$source.s"
    riscv64-elf-gcc -march=rv64imc -mabi=lp64 \
        -c "build/$source.s" -o "build/$source.o"
done
riscv64-elf-gcc -march=rv64imc -mabi=lp64 -c build/start.S -o build/start.o
riscv64-elf-gcc -march=rv64imc -mabi=lp64 -nostdlib -nostartfiles \
    -Wl,--no-relax,-Map=build/linear.map -T build/link.ld \
    build/start.o build/kernel.o build/main.o -o build/linear.elf \
    > build/link.log 2>&1
riscv64-elf-objdump -d -M no-aliases build/linear.elf > build/linear.disasm
riscv64-elf-nm -n -S --defined-only build/linear.elf > build/symbols.txt
riscv64-elf-readelf -h -A -lW build/linear.elf > build/elf-info.txt
spike --isa=RV64IMC -m32 -l --log-commits --log=build/spike.log build/linear.elf
printf 'PASS: standard Spike, model-derived INT8 Linear tile and requantization.\n'
"$PYTHON" profile.py
shasum -a 256 prepare.py kernel.h kernel.c main.c profile.py run.sh \
    build/vectors.h build/start.S build/link.ld > build/source-sha256.txt
