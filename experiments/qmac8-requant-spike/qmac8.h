#ifndef PROJECT_QMAC8_H
#define PROJECT_QMAC8_H
#include <stdint.h>

/* rd is a read/write operand. A one-instruction asm reads all sources before
 * writeback; register overlap is legal. No memory access inside this instruction. */
__attribute__((always_inline)) static inline uint64_t
qmac8_bits(uint64_t old_acc, uint64_t packed_a, uint64_t packed_w) {
    __asm__ volatile(".insn r 0x0b, 0, 0x20, %0, %1, %2"
        : "+r"(old_acc) : "r"(packed_a), "r"(packed_w));
    return old_acc;
}

__attribute__((always_inline)) static inline int32_t
qmac8_acc(int32_t acc, uint64_t packed_a, uint64_t packed_w) {
    uint32_t low = (uint32_t)qmac8_bits((uint64_t)(int64_t)acc,packed_a,packed_w);
    int32_t result;
    __builtin_memcpy(&result,&low,sizeof(result));
    return result;
}
#endif
