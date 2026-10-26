.section .text.init
.globl _start
_start:
    la sp, _stack_top

    # Establish the zero-initialized storage expected by C.
    la t0, _bss_start
    la t1, _bss_end
1:
    bgeu t0, t1, 2f
    sb zero, 0(t0)
    addi t0, t0, 1
    j 1b
2:
    call main

    # main returns the dot product in a0; check before reporting success.
    li t2, 100
    li t0, 3
    bne a0, t2, finish
    li t0, 1
finish:
    la t1, tohost
    sd t0, 0(t1)
hang:
    j hang

.section .tohost, "aw", @progbits
.balign 64
.globl tohost
tohost:
    .dword 0
.balign 64
.globl fromhost
fromhost:
    .dword 0

.section .note.GNU-stack,"",@progbits
