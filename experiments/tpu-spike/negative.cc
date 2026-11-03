#include "tile.h"
#ifndef CASE
#define CASE 0
#endif
alignas(8) static int8_t x[8],w[8];
alignas(8) static int32_t out[64];
int main(){
  project::Tile d{uint64_t(x),uint64_t(w),uint64_t(out),8,8,8,1,1};
  if(CASE==1)d.rows=0;
  if(CASE==2)d.cols=9;
  if(CASE==3)d.k=1025;
  if(CASE==4)d.xstride=7;
  if(CASE==5)d.out++;
  if(CASE==6){asm volatile(".word 0x4400100b");return 0;}
  if(CASE==7){asm volatile(".insn r 0x0b, 0, 0x22, x0, %0, x0" :: "r"(uint64_t(&d)+1):"memory");return 0;}
  if(CASE==8){asm volatile(".insn r 0x0b, 0, 0x22, x1, %0, x0" :: "r"(&d):"memory");return 0;}
  project::tile_dot(d);return 0;
}
