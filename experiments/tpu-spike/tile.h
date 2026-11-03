#ifndef PROJECT_TPU_TILE_H
#define PROJECT_TPU_TILE_H
#include <cstdint>
#include <cstring>
namespace project {
// Fixed 64-byte RV64 little-endian descriptor. Result is dense [rows,cols].
struct alignas(8) Tile { uint64_t x,w,out,k,xstride,wstride,rows,cols; };
static_assert(sizeof(Tile)==64,"descriptor ABI");
inline void tile_dot(const Tile& d) {
#ifdef __riscv
  asm volatile(".insn r 0x0b, 0, 0x22, x0, %0, x0" :: "r"(&d) : "memory");
#else
  auto x=reinterpret_cast<const int8_t*>(d.x),w=reinterpret_cast<const int8_t*>(d.w);
  auto out=reinterpret_cast<int32_t*>(d.out);
  for (unsigned r=0;r<d.rows;++r) for(unsigned c=0;c<d.cols;++c) {
    uint32_t a=0;
    for(unsigned i=0;i<d.k;++i) a+=uint32_t(int(x[r*d.xstride+i])*int(w[c*d.wstride+i]));
    std::memcpy(out+r*d.cols+c,&a,4);
  }
#endif
}
}
#endif
