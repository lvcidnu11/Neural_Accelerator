#include "kernel.h"
#include "tail-vectors.h"
static int32_t actual_acc[6];
static int8_t actual_y[6];
static const int32_t mult[3]={1073741824,1073741824,1073741824};
static const int32_t shift[3]={-3,-3,-3};
volatile int failed_tail_case = -1;
int main(void) {
    for (unsigned c=0;c<sizeof(tail_cases)/sizeof(tail_cases[0]);++c) {
        linear_int8(tail_cases[c].x,tail_cases[c].w,tail_cases[c].bias,mult,shift,
            actual_acc,actual_y,tail_cases[c].m,tail_cases[c].k,tail_cases[c].n);
        for (unsigned i=0;i<6;++i) {
            if (actual_acc[i]!=tail_cases[c].acc[i] || actual_y[i]!=tail_cases[c].y[i]) {
                failed_tail_case=(int)c;
                return 1;
            }
        }
    }
    return 0;
}
