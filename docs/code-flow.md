# Reading and building the project

Start with `experiments/tflm-linear/linear_kernel.h`. `linear<4>` computes four
output channels for one input row at a time. The inner K loop packs eight INT8
pairs per qmac8, while the outer loops cover tokens and channels. The same
kernel can calculate FC1, FC2, QKV and other Linear shapes; FC1 is the main
incremental benchmark, not the only full-model call site.

Follow these files in order:

1. `deit-tiny-inspect/download_weights.py` verifies the official checkpoint;
   `inspect_model.py` checks shapes and creates the float fixture.
2. `deit-linear-int8/prepare.py` creates a simple independent integer Linear
   reference. `tflm-linear/export_model.py` packages FC/GELU graphs and expected
   outputs; `deit-full-int8/export_model.py` builds the complete project graph
   and NumPy integer reference.
3. `tflm-linear/main.cc` initializes TFLM, registers operators, allocates static
   tensors and calls `Invoke()`. `fc_adapter.cc` selects upstream scalar or the
   project kernel. `gelu_lut.cc` implements standalone and fused LUT behavior.
4. `deit-full-int8/main.cc` runs the full graph; `model.cc` implements its three
   custom node types and calls the shared kernel for different matrix shapes.
5. `qmac8-requant-spike/{qmac8,requant8}.h` emits `.insn` on RISC-V. Native host
   checks emulate the arithmetic, but real target verification uses Spike's
   registered handlers in `extension.cc`.
6. `build.py` compiles C++ plus startup, generates `.o`, and links the model,
   TFLM library and `link.ld` into ELF. `start.S` supplies the bare-metal entry
   point, stack/BSS/constructor setup and host termination.
7. `profile.py` checks correctness and measures debugger-delimited retired
   instructions; generated maps/disassembly connect C++ code to executed ISA.

```
checkpoint + synthetic input
  -> Python conversion/reference -> model.tflite + vectors/constants
  -> C++ kernels + TFLM + startup -> object files -> linker -> program.elf
  -> Spike + custom ISA plugin -> correctness + retired-instruction profile
```

A `.tflite` file is serialized graph/data, not C source. TFLM interprets its
operator metadata while calling statically compiled C++ implementations.
Embedding those bytes in the ELF avoids a target filesystem. All conversion
and floating-point reference work runs offline on the host.
