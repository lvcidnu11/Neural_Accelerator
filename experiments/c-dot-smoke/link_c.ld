OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 32M
}
SECTIONS
{
    .text : { *(.text.init) *(.text .text.*) } > RAM
    .rodata : ALIGN(8) { *(.rodata .rodata.*) *(.srodata .srodata.*) } > RAM
    .data : ALIGN(8) { *(.data .data.*) *(.sdata .sdata.*) } > RAM
    .tohost : ALIGN(64) { *(.tohost) } > RAM
    .bss (NOLOAD) : ALIGN(16)
    {
        _bss_start = .;
        *(.bss .bss.*) *(.sbss .sbss.*) *(COMMON)
        _bss_end = .;
    } > RAM
    _stack_top = ORIGIN(RAM) + LENGTH(RAM);
    ASSERT(_bss_end <= _stack_top - 4096, "Program overlaps reserved stack")
}
