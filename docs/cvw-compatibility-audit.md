# CV-Wally compatibility audit, 2026-11-03

## Finding

The existing TPU branch cannot run its custom command directly on the inspected,
unmodified CV-Wally checkout. Its 214,329,448 instruction result is a functional
Spike experiment. There is no implemented tile RTL, custom decode, memory-request
adapter or integrated-core verification for that command. No CV-Wally speedup,
CPI, clock frequency or physical area result exists for it.

The same core-integration gap applies to qmac8/requant8. qmac8 has a standalone
RTL experiment in experiments/qmac-design, but that is not a CV-Wally integration.
The repository does not yet contain requant8 core RTL. The CPU design therefore
also remains an ISA/software result rather than an already runnable Wally core.

This audit does not establish that a tile engine is impossible to integrate.
It establishes that the currently measured functional implementation has not
completed that integration. The engineering feasibility and performance/cost
tradeoff must be demonstrated rather than assumed.

## Direct experiment against the original decoder

Local CV-Wally revision: 2f82f001139358be7646e910ff6fa2a140bed620.
The test compiles the original src/ieu/controller.sv with its original cvw
package, flop and mux dependencies. It uses a minimal RV64IMC+CSR parameter
struct to exercise the decoder. No upstream source is copied or modified.

Verilator 5.050 reports:

| Instruction | Word | IllegalBaseInstrD |
|---|---|---:|
| RV64 ADDI | 0x00000013 | 0 |
| RV64 MUL | 0x02000033 | 0 |
| RV64 LD | 0x00003003 | 0 |
| qmac8 | 0x4005050b | 1 |
| requant8 | 0x4205050b | 1 |
| xtpu, descriptor in a0 | 0x4405000b | 1 |

This is an RTL component test of the actual decoder, not a full-core ELF run
or demonstration of trap delivery/retirement. It independently confirms the
source finding: custom-0 has no implementation in this decoder and retains
the illegal default. Original source hashes and exact commands are recorded
in results/cvw-compatibility-2026-11-03/ and the raw build logs.

## What the Spike plugin implements

experiments/tpu-spike/extension.cc interprets a descriptor, performs multiple
MMU loads, executes nested host C++ multiply/add loops, then performs MMU
stores before returning pc+4. This establishes arithmetic and command semantics.
It does not instantiate a hardware array, model bus arbitration or stall the
simulated CPU for each accelerator cycle. Its separate timing counters are
analytical assumptions, not measurements of the inspected Wally.

Thus the instruction-count reductions remain real ISA-level observations, but
collapsing many products and transfers into one command cannot by itself show
that a practical accelerator is faster, smaller, or energy-efficient.

## Concrete integration gaps

1. Configuration and program layout. The checkout has no rv64imc config. rv64i
   lacks M and compressed support. rv64gc enables M/C but also F/D and other
   hardware, so it is not the intended area baseline. The Spike ELF reserves
   32 MiB, whereas the rv64i DTIM/IROM windows cover 8 MiB. Choose a consistent
   RV64IMC configuration and linker/startup layout before a kernel test.
2. Decode, pipeline ownership and retirement. Decode xtpu, snapshot its
   descriptor pointer, start the engine, hold the appropriate pipeline stages,
   and retire once only after completion. Define forwarding, reset, flush,
   interrupt and fault behavior for a long-running instruction.
3. Tile datapath and local storage. Implement the INT8 MAC array, INT32
   accumulators and operand buffers. The current maximum storage budget is
   two 8x1024-byte buffers plus 64 INT32 accumulators, before control and ports.
   Simultaneous operand supply needs banking/ports and an actual clocked schedule.
4. Memory integration. Either arbitrate requests through the LSU/data-cache
   path or implement a separately managed bus master/scratchpad interface.
   The former needs a request sequencer and fault attribution; the latter
   needs explicit coherence or an uncached memory contract. The existing
   volatile compiler memory clobber does not provide hardware cache coherence.
5. Width and latency. The inspected rv64gc config has AHBW=XLEN=64. Its shared
   external AHB path therefore carries at most 8 payload bytes per accepted
   beat, before wait states/arbitration. The old 16 B/cycle transfer assumption
   is not established for this path. Local banks could still supply 16 B/cycle
   to MACs after staging, but that does not establish external transfer bandwidth.
6. qmac/requant integration too. qmac8 reads old rd plus rs1 and rs2, so the
   core must handle the implicit third source and its hazards. Requantization
   needs the specified wide multiply, rounding and clamp datapath or a defined
   multicycle implementation. Standalone arithmetic tests do not cover these
   core interactions.

7. Fault/retry contract. A code-level inference is that retrying after partial
   stores is not generally idempotent when the output aliases an operand or
   descriptor: the first attempt may overwrite data that a retry reloads.
   This specific alias/fault combination has not been executed in this audit.
   An initial hardware contract should forbid such overlap, or preserve staged
   operands/progress and define restart behavior. The tested model's usual
   separated buffers do not validate these general memory-fault cases.

## Required evidence before a Wally claim

- Run the unchanged scalar control on the selected RV64IMC RTL configuration.
- Integrate and verify qmac8/requant8 on a representative kernel with correct
  source overlap, forwarding, stalls, retirement and output checks.
- Implement the tile engine and request/response adapter, including random
  memory wait states, tails/alignment and fault behavior.
- Execute an actual custom-command kernel on the full modified Wally core and
  compare all outputs with software. Repeat with relevant cache/memory settings.
- Measure kernel cycles, CPI, critical path/Fmax and SAED-mapped area under
  held-constant conditions. Use those results to decide whether to keep TPU.

Until these checks pass, the five-minute deck must call TPU a Spike-only
architectural exploration. The main result is verified full-model ISA/software
behavior. TPU should not be presented as a proven CV-Wally acceleration.

## Reproduce the decoder test

```sh
python3 experiments/cvw-compatibility-2026-11-03/probe.py --wally /path/to/cvw
```

The script does not modify the Wally checkout. It generates a local harness,
compiles original source dependencies into build/, and saves evidence.json.
It requires matching local Verilator and a C++ toolchain. No full Wally RTL
simulation or accelerator implementation is implied by this test.

Primary local references: src/ieu/controller.sv (main decoder/default illegal),
src/lsu/lsu.sv (memory requests, stalls and faults), src/ebu/ebu.sv (IFU/LSU
AHB arbitration), config/rv64i/config.vh and config/rv64gc/config.vh.
AI assistance: Codex performed this audit, generated the decoder harness and
helped qualify presentation claims. Hardware viability remains unproven.
