.section .rodata
array_a:
    .byte -1, 3, 5, 7
array_w:
    .byte 2, 4, 6, 8

.section .text.init
.globl _start
_start:
    la a0, array_a
    la a1, array_w

    li t2, 0
    li t3, 4
loop:
    lb t0, 0(a0)
    lb t1, 0(a1)

    mul t4, t0, t1  # A[i] x W[i]
    add t2, t2, t4  # accumulate
    
    addi a0, a0, 1
    addi a1, a1, 1
    addi t3, t3, -1

    bne t3, zero, loop

    li t3, 96
    bne t2, t3, fail
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

