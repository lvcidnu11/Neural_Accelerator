#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p results
python3 --version > results/python-version.txt
iverilog -V > results/iverilog-version.txt 2>&1
verilator --version > results/verilator-version.txt
yosys -V > results/yosys-version.txt
python3 generate_vectors.py
python3 compare_semantics.py
shasum -a 256 qmac8_unit.sv tb.sv generate_vectors.py compare_semantics.py report.py run.sh abc-generic.script vectors.mem > results/source-sha256.txt
iverilog -g2012 -Wall -s tb -o results/tb.vvp qmac8_unit.sv tb.sv \
    > results/compile.log 2>&1
vvp results/tb.vvp > results/simulation.log
cat results/simulation.log
verilator --lint-only --timing -Wall --top-module tb \
    qmac8_unit.sv tb.sv > results/lint.log 2>&1
for lanes in 1 2 8; do
    yosys -Q -T -p "read_verilog -sv qmac8_unit.sv; chparam -set LANES $lanes qmac8_unit; hierarchy -top qmac8_unit; proc; opt; wreduce; opt; stat; synth -noabc -top qmac8_unit; abc -script abc-generic.script; clean; check -assert; stat; write_json results/lanes$lanes.json" \
        > "results/synthesis-lanes$lanes.log" 2>&1
done
python3 report.py
printf 'Generic synthesis completed. These cell counts are not SAED area or Fmax.\n'
