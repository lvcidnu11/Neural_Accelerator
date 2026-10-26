#include "requant8.h"
#include "kernel.h"
#include "requant-vectors.h"
volatile int failed_requant_case=-1;

static uint64_t pattern(unsigned mode,uint64_t x,uint64_t params) {
    register uint64_t r0 __asm__("t0")=x;
    register uint64_t r1 __asm__("t1")=params;
    register uint64_t r2 __asm__("t2")=UINT64_C(0x123456789abcdef0);
#define EXEC(text) __asm__ volatile(text : "+r"(r0),"+r"(r1),"+r"(r2))
    switch(mode) {
    case 1: EXEC(".insn r 0x0b,0,0x21,%0,%0,%1"); return r0;
    case 2: EXEC(".insn r 0x0b,0,0x21,%1,%0,%1"); return r1;
    case 3: EXEC(".insn r 0x0b,0,0x21,%2,%1,%1"); return r2;
    case 4: EXEC(".insn r 0x0b,0,0x21,%1,%1,%1"); return r1;
    case 5: EXEC(".insn r 0x0b,0,0x21,%2,zero,%1"); return r2;
    case 6: EXEC(".insn r 0x0b,0,0x21,%2,%0,zero"); return r2;
    case 7: EXEC(".insn r 0x0b,0,0x21,zero,%0,%1\nmv %2,zero"); return r2;
    case 8: EXEC(".insn r 0x0b,0,0x21,zero,zero,zero\nmv %2,zero"); return r2;
    default: return requant8_bits(x,params);
    }
#undef EXEC
}

int main(void) {
    for (unsigned i=0;i<sizeof(scalar_requant_cases)/sizeof(scalar_requant_cases[0]);++i) {
        int8_t software=clamp_int8(multiply_quantized(scalar_requant_cases[i].x,
            scalar_requant_cases[i].m,scalar_requant_cases[i].shift));
        int8_t custom=requant8_int8(scalar_requant_cases[i].x,
            requant8_params(scalar_requant_cases[i].m,scalar_requant_cases[i].shift,0));
        if (software != scalar_requant_cases[i].expected || software != custom) {
            failed_requant_case=(int)i;
            return 1;
        }
    }
    for (unsigned i=0;i<sizeof(requant_instruction_cases)/sizeof(requant_instruction_cases[0]);++i) {
        uint64_t output=pattern(requant_instruction_cases[i].mode,
            requant_instruction_cases[i].x,requant_instruction_cases[i].params);
        if (output != requant_instruction_cases[i].expected) {
            failed_requant_case=(int)i;
            return 2;
        }
    }
    return 0;
}
