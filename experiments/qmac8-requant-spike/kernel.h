#ifndef DEIT_LINEAR_KERNEL_H
#define DEIT_LINEAR_KERNEL_H
#include <stdint.h>
/* Preconditions: m>=0,k>=0,n>=0; buffers valid; every accumulator partial sum
 * fits INT32; multiplier>=0, shift in [-31,30], scaled result fits INT32. */
int32_t multiply_quantized(int32_t x, int32_t multiplier, int shift);
int8_t clamp_int8(int32_t x);
void linear_int8(const int8_t *x, const int8_t *w, const int32_t *bias,
                 const int32_t *multipliers, const int32_t *shifts,
                 int32_t *acc_out, int8_t *output, int m, int k, int n);
#endif
