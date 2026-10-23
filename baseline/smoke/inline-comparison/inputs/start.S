.section .text.init
.globl _start

_start:
    la sp, _stack_top

    call main

    li t3, 100
    bne a0, t3, fail

pass:
    li t0, 1
    j finish

fail:
    li t0, 3

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
