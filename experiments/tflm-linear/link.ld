OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 32M }
SECTIONS
{
  .text : { KEEP(*(.text.init)) *(.text .text.*) } > RAM
  .rodata : ALIGN(16) { *(.rodata .rodata.*) *(.srodata .srodata.*) } > RAM
  .data : ALIGN(16) { *(.data .data.*) *(.sdata .sdata.*) } > RAM
  .init_array : ALIGN(8) {
    __init_array_start = .;
    KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*))) KEEP(*(.init_array))
    __init_array_end = .;
  } > RAM
  .tohost : ALIGN(64) { KEEP(*(.tohost)) } > RAM
  .bss (NOLOAD) : ALIGN(16) {
    _bss_start = .;
    *(.bss .bss.*) *(.sbss .sbss.*) *(COMMON)
    _bss_end = .;
  } > RAM
  _stack_top = ORIGIN(RAM) + LENGTH(RAM);
  ASSERT(_bss_end <= _stack_top - 65536, "Program overlaps reserved stack")
}
