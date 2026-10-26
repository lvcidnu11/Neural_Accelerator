#include "qmac8.h"
#include "instruction-vectors.h"
volatile int failed_instruction_case = -1;

static uint64_t pattern(unsigned mode,uint64_t old,uint64_t a,uint64_t w) {
    register uint64_t r0 __asm__("t0") = old;
    register uint64_t r1 __asm__("t1") = a;
    register uint64_t r2 __asm__("t2") = w;
#define EXEC(text) __asm__ volatile(text : "+r"(r0), "+r"(r1), "+r"(r2))
    switch(mode) {
    case 1: EXEC(".insn r 0x0b,0,0x20,%0,%0,%2"); break;
    case 2: EXEC(".insn r 0x0b,0,0x20,%0,%1,%0"); break;
    case 3: EXEC(".insn r 0x0b,0,0x20,%0,%0,%0"); break;
    case 4: EXEC(".insn r 0x0b,0,0x20,%0,%1,%1"); break;
    case 5: EXEC(".insn r 0x0b,0,0x20,zero,%1,%2\nmv %0,zero"); break;
    case 6: EXEC(".insn r 0x0b,0,0x20,%0,zero,%2"); break;
    case 7: EXEC(".insn r 0x0b,0,0x20,%0,%1,zero"); break;
    case 8: EXEC(".insn r 0x0b,0,0x20,%0,zero,zero"); break;
    default: return qmac8_bits(old,a,w);
    }
#undef EXEC
    return r0;
}

int main(void) {
    for (unsigned i=0;i<sizeof(instruction_cases)/sizeof(instruction_cases[0]);++i) {
        uint64_t result=pattern(instruction_cases[i].mode,instruction_cases[i].old,
            instruction_cases[i].a,instruction_cases[i].w);
        if (result != instruction_cases[i].expected) {
            failed_instruction_case=(int)i;
            return 1;
        }
    }
    uint64_t acc=UINT64_C(0x7fffffff);
    for (unsigned i=0;i<33;++i)
        acc=qmac8_bits(acc,UINT64_C(0x8080808080808080),UINT64_C(0x8080808080808080));
    return acc==UINT64_C(0xffffffff8041ffff) ? 0 : 2;
}
