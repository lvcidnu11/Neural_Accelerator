#ifndef PROJECT_LINEAR_KERNEL_H
#define PROJECT_LINEAR_KERNEL_H
#include <cstdint>
#include <cstring>
#if defined(__riscv) && !defined(PROJECT_SCALAR_BASELINE)
#include "../qmac8-requant-spike/qmac8.h"
#include "../qmac8-requant-spike/requant8.h"
#endif

#ifdef USE_TPU
#include "../tpu-spike/tile.h"
#endif

namespace project {
inline uint64_t mac(uint64_t acc,uint64_t a,uint64_t b) {
#if defined(__riscv) && !defined(PROJECT_SCALAR_BASELINE)
  return qmac8_bits(acc,a,b);
#else
  uint32_t low=static_cast<uint32_t>(acc);
  for (unsigned i=0;i<8;++i) {
    int x=static_cast<int>((a>>(8*i))&255),w=static_cast<int>((b>>(8*i))&255);
    if (x>=128) x-=256;
    if (w>=128) w-=256;
    low+=static_cast<uint32_t>(x*w);
  }
  int32_t signed_low; std::memcpy(&signed_low,&low,4);
  return static_cast<uint64_t>(static_cast<int64_t>(signed_low));
#endif
}
inline uint64_t load8(const int8_t* p) {
  uint64_t x; std::memcpy(&x,__builtin_assume_aligned(p,8),8); return x;
}
inline int8_t requant(int32_t acc,int32_t multiplier,int shift) {
#if defined(__riscv) && !defined(PROJECT_SCALAR_BASELINE)
  return requant8_int8(acc,requant8_params(multiplier,shift,0));
#else
  int bits=31-shift;
  int64_t v=static_cast<int64_t>(acc)*multiplier+(INT64_C(1)<<(bits-1));
  int64_t scaled=v>=0 ? (v>>bits) : -(((-v)+((INT64_C(1)<<bits)-1))>>bits);
  return static_cast<int8_t>(scaled>127 ? 127 : (scaled< -128 ? -128 : scaled));
#endif
}
inline int32_t low32(uint64_t v) {
  uint32_t u=static_cast<uint32_t>(v); int32_t s; std::memcpy(&s,&u,4); return s;
}
inline int8_t finish(uint64_t acc,int32_t multiplier,int shift,const int8_t* lut) {
  int8_t q=requant(low32(acc),multiplier,shift);
  return lut ? lut[static_cast<int>(q)+128] : q;
}

// Requires proven INT32 accumulator bounds, symmetric zero points, full INT8
// output clamp. The table preserves the exact INT8 boundary before GELU.
#ifdef PROJECT_LINEAR_INLINE
#define PROJECT_LINEAR_ATTR __attribute__((always_inline)) inline
#else
#define PROJECT_LINEAR_ATTR __attribute__((noinline))
#endif
template<int B> PROJECT_LINEAR_ATTR void linear(
    const int8_t* x,const int8_t* w,const int32_t* bias,const int32_t* mult,
    const int32_t* shift,int8_t* output,int m,int k,int n,const int8_t* lut=nullptr) {
  static_assert(B==1 || B==2 || B==4,"supported blocking factors");
#ifdef PROJECT_SCALAR_BASELINE
  // Direct scalar multiply/add, not software emulation of a packed instruction.
  // Keep B-channel input reuse and the same quantization / LUT boundary.
  for(int row=0;row<m;++row) {
    int c=0;
    for(;c+B<=n;c+=B) {
      const int8_t* xp=x+row*k;
      const int8_t* w0=w+c*k;
      uint32_t a0=uint32_t(bias[c]),a1=0,a2=0,a3=0;
      if constexpr(B>=2) a1=uint32_t(bias[c+1]);
      if constexpr(B==4){a2=uint32_t(bias[c+2]);a3=uint32_t(bias[c+3]);}
      for(int i=0;i<k;++i) {
        int v=xp[i];a0+=uint32_t(v*int(w0[i]));
        if constexpr(B>=2) a1+=uint32_t(v*int(w0[k+i]));
        if constexpr(B==4){a2+=uint32_t(v*int(w0[2*k+i]));a3+=uint32_t(v*int(w0[3*k+i]));}
      }
      output[row*n+c]=finish(a0,mult[c],shift[c],lut);
      if constexpr(B>=2) output[row*n+c+1]=finish(a1,mult[c+1],shift[c+1],lut);
      if constexpr(B==4){output[row*n+c+2]=finish(a2,mult[c+2],shift[c+2],lut);output[row*n+c+3]=finish(a3,mult[c+3],shift[c+3],lut);}
    }
    for(;c<n;++c){
      uint32_t acc=uint32_t(bias[c]);for(int i=0;i<k;++i)acc+=uint32_t(int(x[row*k+i])*int(w[c*k+i]));
      output[row*n+c]=finish(acc,mult[c],shift[c],lut);
    }
  }
  return;
#endif

#ifdef USE_TPU
  // CPU keeps bias, requantization and optional GELU identical to the baseline.
  for (int row=0;row<m;row+=8) for(int c=0;c<n;c+=8) {
    int rows=m-row<8 ? m-row : 8,cols=n-c<8 ? n-c : 8;
    alignas(8) int32_t sums[64];
    Tile d{reinterpret_cast<uint64_t>(x+row*k),reinterpret_cast<uint64_t>(w+c*k),
      reinterpret_cast<uint64_t>(sums),uint64_t(k),uint64_t(k),uint64_t(k),uint64_t(rows),uint64_t(cols)};
    tile_dot(d);
#ifdef TPU_HOIST_PARAMS
    for(int j=0;j<cols;++j) {
      const uint32_t bv=uint32_t(bias[c+j]);
      const int32_t multiplier=mult[c+j];
      const int scale_shift=shift[c+j];
#if defined(__riscv) && !defined(PROJECT_SCALAR_BASELINE)
      const uint64_t params=requant8_params(multiplier,scale_shift,0);
#endif
      for(int r=0;r<rows;++r) {
        const uint32_t bits=uint32_t(sums[r*cols+j])+bv;
#if defined(__riscv) && !defined(PROJECT_SCALAR_BASELINE)
        int8_t q=requant8_int8(low32(bits),params);
#else
        int8_t q=requant(low32(bits),multiplier,scale_shift);
#endif
        output[(row+r)*n+c+j]=lut ? lut[int(q)+128] : q;
      }
    }
#else
    for(int r=0;r<rows;++r) for(int j=0;j<cols;++j) {
      uint32_t bits=uint32_t(sums[r*cols+j])+uint32_t(bias[c+j]);
      output[(row+r)*n+c+j]=finish(bits,mult[c+j],shift[c+j],lut);
    }
#endif
  }
  return;
#endif

  for (int row=0;row<m;++row) {
    int c=0;
    for (;c+B<=n;c+=B) {
      const int8_t *xp=x+row*k,*w0=w+c*k,*w1=w0,*w2=w0,*w3=w0;
      if constexpr(B>=2) w1=w0+k;
      if constexpr(B==4) { w2=w1+k; w3=w2+k; }
      uint64_t a0=static_cast<uint64_t>(static_cast<int64_t>(bias[c]));
      uint64_t a1=0,a2=0,a3=0;
      if constexpr(B>=2) a1=static_cast<uint64_t>(static_cast<int64_t>(bias[c+1]));
      if constexpr(B==4) {
        a2=static_cast<uint64_t>(static_cast<int64_t>(bias[c+2]));
        a3=static_cast<uint64_t>(static_cast<int64_t>(bias[c+3]));
      }
      int i=0;
      if (((reinterpret_cast<uintptr_t>(xp)|reinterpret_cast<uintptr_t>(w0)|
            reinterpret_cast<uintptr_t>(w1)|reinterpret_cast<uintptr_t>(w2)|
            reinterpret_cast<uintptr_t>(w3))&7)==0) {
        for (;k-i>=8;i+=8) {
          uint64_t a=load8(xp+i);
          a0=mac(a0,a,load8(w0+i));
          if constexpr(B>=2) a1=mac(a1,a,load8(w1+i));
          if constexpr(B==4) {
            a2=mac(a2,a,load8(w2+i)); a3=mac(a3,a,load8(w3+i));
          }
        }
      }
      // Unsigned addition preserves the ISA modulo semantics without C UB.
      for (;i<k;++i) {
        int v=xp[i]; a0+=static_cast<uint32_t>(v*int(w0[i]));
        if constexpr(B>=2) a1+=static_cast<uint32_t>(v*int(w1[i]));
        if constexpr(B==4) { a2+=static_cast<uint32_t>(v*int(w2[i])); a3+=static_cast<uint32_t>(v*int(w3[i])); }
      }
      output[row*n+c]=finish(a0,mult[c],shift[c],lut);
      if constexpr(B>=2) output[row*n+c+1]=finish(a1,mult[c+1],shift[c+1],lut);
      if constexpr(B==4) {
        output[row*n+c+2]=finish(a2,mult[c+2],shift[c+2],lut);
        output[row*n+c+3]=finish(a3,mult[c+3],shift[c+3],lut);
      }
    }
    for (;c<n;++c) {
      uint64_t a=static_cast<uint64_t>(static_cast<int64_t>(bias[c]));
      for (int i=0;i<k;++i) a+=static_cast<uint32_t>(int(x[row*k+i])*int(w[c*k+i]));
      output[row*n+c]=finish(a,mult[c],shift[c],lut);
    }
  }
}
#undef PROJECT_LINEAR_ATTR
} // namespace project
#endif
