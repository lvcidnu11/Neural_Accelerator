#include <initializer_list>
#include "linear_kernel.h"
alignas(16) static int8_t x[96],w[320],output[50],lut[256];
static int32_t bias[9],mult[9],shift[9];
static uint32_t state=6455;
static int8_t random_int8() { state=state*1664525u+1013904223u; return int8_t((state>>24)-128); }
extern "C" volatile unsigned test_cases=0;
int main() {
  for (int i=0;i<256;++i) lut[i]=int8_t((i*37)%256-128);
  for (auto& v:x) v=random_int8(); for (auto& v:w) v=random_int8();
  for (int i=0;i<9;++i) { bias[i]=i*77-300;mult[i]=(1<<30)+i*17;shift[i]=-7; }
  for (int k: {0,1,7,8,9,15,16,17,31,32})
    for (int n: {1,2,3,4,5,7,8,9})
      for (int offset: {0,1}) for (int b: {2,4}) for (int fuse: {0,1}) {
        for (auto& v:output) v=99;
        const int8_t* table=fuse ? lut : nullptr;
        if (b==2) project::linear<2>(x+offset,w+offset,bias,mult,shift,output,2,k,n,table);
        else project::linear<4>(x+offset,w+offset,bias,mult,shift,output,2,k,n,table);
        for (int row=0;row<2;++row) for (int c=0;c<n;++c) {
          int64_t acc=bias[c];
          for (int i=0;i<k;++i) acc+=int64_t(x[offset+row*k+i])*w[offset+c*k+i];
          int64_t v=acc*mult[c]+(INT64_C(1)<<37);
          const int64_t divisor=INT64_C(1)<<38;
          int64_t q=v/divisor; if (v<0 && v%divisor) --q;
          if (q>127) q=127; if (q< -128) q= -128;
          int8_t expected=table ? table[q+128] : int8_t(q);
          if (output[row*n+c]!=expected) return 1;
        }
        for (int i=2*n;i<50;++i) if (output[i]!=99) return 2;
        test_cases=test_cases+1;
      }
  return test_cases==640 ? 0 : 3;
}
