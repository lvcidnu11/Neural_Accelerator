.section .text.init
.globl _start
_start:
    la sp, _stack_top
    la t0, _bss_start
    la t1, _bss_end
1:
    bgeu t0, t1, 2f
    sb zero, 0(t0)
    addi t0, t0, 1
    j 1b
2:
    la s0, __init_array_start
    la s1, __init_array_end
3:
    bgeu s0, s1, 4f
    ld t0, 0(s0)
    addi s0, s0, 8
    jalr t0
    j 3b
4:
    call main
.globl test_finished
test_finished:
    slli t0, a0, 1
    ori t0, t0, 1
    la t1, tohost
    sd t0, 0(t1)
5:
    j 5b
.section .tohost,"aw",@progbits
.balign 64
.globl tohost
tohost: .dword 0
.balign 64
.globl fromhost
fromhost: .dword 0
.section .note.GNU-stack,"",@progbits
