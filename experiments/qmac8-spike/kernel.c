#include "qmac8.h"
#include "kernel.h"

/* Integer-only form of TFLite v2.19.0 single-rounding scaling.
 * Floor division is expressed without relying on signed right-shift behavior.
 * Preconditions in kernel.h are verified by the fixture generator. */
int32_t multiply_quantized(int32_t x, int32_t multiplier, int shift) {
    int bits = 31 - shift;
    int64_t rounded = (int64_t)x * multiplier + (INT64_C(1) << (bits - 1));
    if (rounded >= 0)
        return (int32_t)((uint64_t)rounded >> bits);
    uint64_t magnitude = (uint64_t)(-rounded);
    uint64_t ceiling = (magnitude + ((UINT64_C(1) << bits) - 1)) >> bits;
    return (int32_t)(-(int64_t)ceiling);
}

int8_t clamp_int8(int32_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

/* X[M,K], W[N,K], bias[N] -> Y[M,N]. Symmetric zero points = 0.
 * Ordinary scalar RV64IMC baseline. No packed/custom instructions. */
void linear_int8(const int8_t *x, const int8_t *w, const int32_t *bias,
                 const int32_t *multipliers, const int32_t *shifts,
                 int32_t *acc_out, int8_t *output, int m, int k, int n) {
    for (int row = 0; row < m; ++row) {
        for (int channel = 0; channel < n; ++channel) {
            int32_t acc = bias[channel];
#ifdef USE_QMAC8
            const int8_t *xp = x + row*k;
            const int8_t *wp = w + channel*k;
            int i = 0;
            /* A word load is used only when both pointers are 8-byte aligned.
             * Row strides not divisible by eight can select the scalar path. */
            if ((((uintptr_t)xp | (uintptr_t)wp) & 7u) == 0) {
                uint64_t packed_acc = (uint64_t)(int64_t)acc;
                for (; k-i >= 8; i += 8) {
                    uint64_t a, b;
                    __builtin_memcpy(&a,__builtin_assume_aligned(xp+i,8),8);
                    __builtin_memcpy(&b,__builtin_assume_aligned(wp+i,8),8);
                    packed_acc = qmac8_bits(packed_acc,a,b);
                }
                /* Keep the architectural 64-bit result between qmac8s to avoid
                 * asking GCC for a redundant INT32 sign extension each group. */
                uint32_t low = (uint32_t)packed_acc;
                __builtin_memcpy(&acc,&low,sizeof(acc));
            }
            for (; i < k; ++i)
                acc += (int32_t)xp[i] * (int32_t)wp[i];
#else
            for (int i = 0; i < k; ++i)
                acc += (int32_t)x[row*k+i] * (int32_t)w[channel*k+i];
#endif
            acc_out[row*n+channel] = acc;
            output[row*n+channel] = clamp_int8(multiply_quantized(
                acc, multipliers[channel], shifts[channel]));
        }
    }
}
