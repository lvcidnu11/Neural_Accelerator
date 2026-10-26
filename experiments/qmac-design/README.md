# QMAC arithmetic-unit design experiment

Measured 2026-10-23. This folder contains three standalone RTL arithmetic-unit
candidates, **not a working custom ISA instruction**. No instruction encoding,
Spike extension, register-file ports, CV-Wally pipeline integration, or core
stall/flush/forwarding logic has been implemented here. Existing C dot-product
programs remain scalar RV64IMC programs.

## What is an adder tree?

For the existing positive example, eight simultaneous INT8 products are:

```text
products: 2  12   30  56   90  132   182  240
level 1:   14      86       222        422
level 2:       100                644
level 3:                 744
```

Both a serial chain and the tree use seven additions to sum eight products.
The balanced tree has three dependent adder levels, while a plain serial chain
has seven. This is combinational dependency depth, not automatically three
clock cycles. In this experiment the eight-lane combinational reduction and
accumulator add sit between registers without internal pipeline registers.
Physical synthesis may restructure this source-level arithmetic.

## Experimental semantics (not yet a frozen ISA)

Two 64-bit packed inputs, eight signed INT8 lanes each; lane i is bits
[8*i+7:8*i]. A separate 32-bit accumulator input is supplied directly by the
testbench. Result is `(acc_in + sum(a_i*w_i)) mod 2^32`. Interpret returned bits
as signed INT32 if needed. There is no saturation, bias beyond acc_in, zero-point
correction, or requantization. Modulo addition permits regrouping partial sums;
per-step saturating addition would not generally permit the same regrouping.

Each signed product has 16 bits. The eight-way tree grows 16 -> 17 -> 18 -> 19
bits; eight (-128)*(-128) products sum to +131072, which needs signed 19 bits.
Narrowing the final dot to signed 18 bits would lose this positive endpoint.

Inputs are captured on a rising edge when start && !busy. While busy,
new requests are ignored. done pulses when the registered output is ready.
Synchronous reset cancels pending work. All candidates latch the same inputs
and use the same result interface. All three include input/control/output
register overhead; cell counts are not just multiplier counts.

## Verification and results

Reproduce:

```sh
cd experiments/qmac-design
sh run.sh
```

Python arbitrary-precision integers supply the independent reference. 1,042
vectors (1,024 random, seed 6455, plus 18 directed) are checked against all
three designs, including the value 744, zero, negative values, INT8 extremes,
individual lane activation, and positive/negative INT32 wraparound. Testbench
also checks exact completion cycles, one-cycle done pulses, changing external
inputs after acceptance, ignored requests during busy, reset during work at
multiple phases, and recovery without stale completions. Icarus simulation
passed; strict Verilator lint passed. Yosys check -assert passed for all three
synthesized variants.

| Candidate | RTL arithmetic multipliers before mapping | Verified result latency | Generic combinational cells | Generic sequential cells | Total generic cells |
|---|---:|---:|---:|---:|---:|
| One lane, reused | 1 | 8 cycles | 851 | 197 | 1,048 |
| Two lanes, reused | 2 | 4 cycles | 1,446 | 197 | 1,643 |
| Eight lanes, balanced tree | 8 | 1 cycle | 4,742 | 197 | 4,939 |

Latency is the rising-edge distance from input acceptance to done. It excludes
the input acceptance edge itself. These wrappers do not accept another request
on the completion edge; a request can be accepted on the next edge. Therefore
the minimum acceptance intervals in this implementation are 9, 5 and 2 cycles,
respectively, rather than equating latency with throughput. Back-to-back peak
throughput was not measured in this test.

The testbench's 10 ns period is only an event schedule and is NOT verified
physical timing. One cycle in this simulation does not establish one cycle at
CV-Wally's target clock, nor eight-fold CPU speedup. Cycle-based comparison
assumes a common feasible clock, which remains to be established.

## Synthesis interpretation

Yosys lowers arithmetic to generic logic and maps with the same explicit ABC
script (`strash; map;`) for all candidates. It has no process/library timing
constraints. The script is saved rather than relying on a changing default
optimization recipe. Generic cell counts show a structural hardware-cost trend;
different gate types have different physical costs. Do not call these counts
area in um^2, transistor counts, or clock frequency. They are not the assignment's
required DC/RTL Architect + SAED32/28 area results. No power was measured.

Packing bytes into lanes is wiring for the full-parallel candidate; serial
candidates need selection logic. None of these units fetches its operands from
memory or models instruction fetch/decode. Input load cost and memory traffic
must be measured in the eventual kernel/core experiment.

## Evidence and next decisions

- qmac8_unit.sv: parameterized candidate arithmetic and control.
- generate_vectors.py, vectors.mem, vectors.json: independent reference inputs.
- tb.sv: directed/random functional and interface timing checks.
- results/simulation.log and lint.log: actual simulation and lint outputs.
- results/synthesis-lanes*.log: arithmetic resource and generic mapping logs.
- results/lanes*.json: generated generic netlists; not physical layouts.
- results/summary.json: machine-readable results from report.py.
- results/*version.txt and source-sha256.txt: tools and source hashes.

Next, choose provisional ISA semantics and decide how old accumulator plus two
packed inputs reach the unit. A three-input unit in isolation does not solve
register-file port or encoding constraints. Midterm work still needs a custom
encoding, Spike execution, software invocation and reference-based measurements.
Later, integrate/verify RTL and obtain real timing/area before choosing a winner.

## One instruction versus two: arithmetic check

`python3 compare_semantics.py` compares provisional register-level arithmetic
models, not actual Spike instruction execution:

```text
one: rd = sign_extend_64(low32(old_rd + dot8(rs1, rs2)))
two: tmp = sign_extend_64(dot8(rs1, rs2)); rd = ADDW(old_rd, tmp)
```

The dot uses signed INT8 lanes and fits signed 19 bits; old_rd contributes its
low 32 bits. Both models matched the independent Python-list reference on all
1,042 vectors, including 2 that mathematically exceed signed INT32 range and
must wrap. Arbitrary non-sign-extended high bits were supplied in old_rd to
check the low32 rule. Output high bits are the sign extension of bit 31.
The directed full-register results are in results/semantics-comparison.json.

Algebraically, low32(acc + dot) is unchanged by taking low32 of either operand
before addition. Since ADDW sign-extends that low32 sum, both paths return the
same 64-bit register value. This equivalence is for the stated wrap semantics,
not permission to change a model's reference quantization or overflow behavior.

This validates the arithmetic comparison only. It does not compare the hardware
area/cycles of a two-read-port DOT implementation against a three-input MAC
implementation: the earlier LANES=1/2/8 table supplied accumulator directly to
every unit and compared multiplier parallelism, not register-file integration.

These candidate RTL/test/experiment files were generated with AI assistance;
review and understand the arithmetic, interface, tests and limitations before
using them in a course submission, subject to the course's AI policy.
