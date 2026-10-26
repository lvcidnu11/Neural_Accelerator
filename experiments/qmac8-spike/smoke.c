#include "qmac8.h"
int main(void) {
    return qmac8_bits(0,UINT64_C(0x0f0d0b0907050301),
                         UINT64_C(0x100e0c0a08060402)) == 744 ? 0 : 1;
}
