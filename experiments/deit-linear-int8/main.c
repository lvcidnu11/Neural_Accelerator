#include "kernel.h"
#include "vectors.h"

static int32_t actual_acc[TEST_M*TEST_N];
static int8_t actual_y[TEST_M*TEST_N];
volatile int32_t failed_index = -1;
volatile int32_t failure_stage = 0;

int main(void) {
    for (unsigned i = 0; i < sizeof(requant_cases)/sizeof(requant_cases[0]); ++i) {
        int32_t scaled = multiply_quantized(requant_cases[i].x,
            requant_cases[i].m, requant_cases[i].shift);
        if (scaled != requant_cases[i].scaled ||
            clamp_int8(scaled) != requant_cases[i].clamped) {
            failure_stage = 1;
            failed_index = (int32_t)i;
            return 1;
        }
    }
#ifdef __riscv
    __asm__ volatile(".globl kernel_begin\nkernel_begin:" ::: "memory");
#endif
    linear_int8(input_q, weight_q, bias_q, multipliers, shifts,
        actual_acc, actual_y, TEST_M, TEST_K, TEST_N);
#ifdef __riscv
    __asm__ volatile(".globl kernel_end\nkernel_end:" ::: "memory");
#endif
    for (int i = 0; i < TEST_M*TEST_N; ++i) {
        if (actual_acc[i] != expected_acc[i] || actual_y[i] != expected_y[i]) {
            failure_stage = 2;
            failed_index = i;
            return 2;
        }
    }
    return 0;
}
