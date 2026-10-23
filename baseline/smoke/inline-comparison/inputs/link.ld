OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 32M
}

SECTIONS
{
    .text :
    {
        *(.text.init)
        *(.text .text.*)
    } > RAM

    .tohost : ALIGN(64)
    {
        *(.tohost)
    } > RAM
    _stack_top = ORIGIN(RAM) + LENGTH(RAM); /* so the stack would be in the RAM */
}
