#include <initializer_list>
#define USE_TPU
#include "../tflm-linear/linear_kernel.h"
alignas(16) static int8_t x[17*1024+1],w[17*1024+1],y[17*17+16],lut[256];
static int32_t bias[17],mult[17],shift[17];
extern "C" { volatile unsigned test_cases=0; }
static uint32_t rng=6455;
static int8_t random8() { rng=rng*1664525u+1013904223;return int8_t(rng>>24); }
int main() {
  for(auto& a:x)a=random8();for(auto& a:w)a=random8();
  for(int i=0;i<256;++i)lut[i]=int8_t((i*37)%256-128);
  for(int i=0;i<17;++i){bias[i]=i*83-600;mult[i]=(1<<30)+17*i;shift[i]=-10;}
#ifdef LOG_SMOKE
  for(int m:{9}) for(int n:{9}) for(int k:{9})
#else
  for(int m:{1,7,8,9,17}) for(int n:{1,7,8,9,17})
  for(int k:{0,1,7,8,9,192,197,768,1024})
#endif
  for(int offset:{0,1}) for(int fuse:{0,1}) {
    for(auto& a:y)a=99;
    project::linear<4>(x+offset,w+offset,bias,mult,shift,y,m,k,n,fuse?lut:nullptr);
    for(int r=0;r<m;++r)for(int c=0;c<n;++c){
      int64_t acc=bias[c];for(int i=0;i<k;++i)acc+=int64_t(x[offset+r*k+i])*w[offset+c*k+i];
      int64_t numerator=acc*mult[c]+(INT64_C(1)<<40),denom=INT64_C(1)<<41;
      int64_t q=numerator/denom;if(numerator<0 && numerator%denom)--q;
      if(q>127)q=127;if(q< -128)q= -128;
      if(y[r*n+c]!=(fuse?lut[q+128]:int8_t(q)))return 1;
    }
    for(int i=m*n;i<int(sizeof(y));++i)if(y[i]!=99)return 2;
    ++test_cases;
  }
  // Directed extrema and modulo accumulator semantics, independent of requant.
  for(auto& a:x)a=-128;for(auto& a:w)a=-128;
  int32_t sums[64];project::Tile d{uint64_t(x),uint64_t(w),uint64_t(sums),1024,1024,1024,8,8};
  project::tile_dot(d);for(auto a:sums)if(a!=16777216)return 3;
  #ifdef LOG_SMOKE
  return test_cases==4?0:4;
#else
  return test_cases==900?0:4;
#endif
}
