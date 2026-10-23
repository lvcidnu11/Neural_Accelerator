#include <stdint.h>

static const int8_t A[4] = {1 ,3, 5, 7};
static const int8_t W[4] = {2, 4, 6, 8};

int32_t dot8_t (const int8_t *a, const int8_t *w, int n) {
    int32_t acc = 0;
    for (int32_t i = 0; i < n; i++) {
        acc += a[i] * w[i];
    }
    return acc;
}

int main() {
    int32_t result = dot8_t(A, W, 4);
    return result;
}