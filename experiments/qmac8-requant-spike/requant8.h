#ifndef PROJECT_REQUANT8_H
#define PROJECT_REQUANT8_H
#include <stdint.h>

/* rs2: multiplier[31:0], signed shift[39:32], signed output zp[47:40].
 * Multiplier must be nonnegative, shift in [-31,30]; reserved[63:48]=0. */
__attribute__((always_inline)) static inline uint64_t
requant8_params(int32_t multiplier,int shift,int zero_point) {
    return (uint64_t)(uint32_t)multiplier |
        ((uint64_t)(uint8_t)shift << 32) |
        ((uint64_t)(uint8_t)zero_point << 40);
}

__attribute__((always_inline)) static inline uint64_t
requant8_bits(uint64_t accumulator,uint64_t params) {
    uint64_t result;
    __asm__ volatile(".insn r 0x0b,0,0x21,%0,%1,%2"
        : "=r"(result) : "r"(accumulator),"r"(params));
    return result;
}

__attribute__((always_inline)) static inline int8_t
requant8_int8(int32_t accumulator,uint64_t params) {
    uint8_t low=(uint8_t)requant8_bits((uint64_t)(int64_t)accumulator,params);
    int8_t result;
    __builtin_memcpy(&result,&low,1);
    return result;
}
#endif
