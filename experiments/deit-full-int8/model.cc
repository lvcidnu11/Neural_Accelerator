#include <cstdint>
#include <cstring>
#include "../tflm-linear/linear_kernel.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace {
constexpr int T=197,D=192,H=768,PAD=200;
alignas(16) int8_t normalized[T*D],qkv[T*3*D],context_data[T*D],tmp[T*D],hidden[T*H];
alignas(16) int8_t queries[T*64],keys[T*64],values[64*PAD],scores[T*T],probs[T*PAD],head_output[T*64];
int32_t multipliers[T],shifts[T],biases[T];

int8_t clip(int64_t x) { return int8_t(x>127 ? 127 : (x< -128 ? -128 : x)); }
int64_t floor_div(int64_t numerator,int64_t divisor) {
  int64_t q=numerator/divisor; if (numerator%divisor<0) --q; return q;
}
int32_t scaled(int32_t x,int32_t m,int32_t shift) {
  int bits=31-shift;
  int64_t numerator=int64_t(x)*m+(INT64_C(1)<<(bits-1));
  // Express floor division without relying on C signed division rounding.
  return int32_t(numerator>=0 ? (numerator>>bits) : -(((-numerator)+((INT64_C(1)<<bits)-1))>>bits));
}
uint64_t isqrt(uint64_t x) {
  uint64_t result=0,bit=UINT64_C(1)<<62;
  while(bit>x) bit>>=2;
  while(bit) { if (x>=result+bit) { x-=result+bit;result=(result>>1)+bit; }
    else result>>=1; bit>>=2; }
  return result;
}
struct Params {
  const uint8_t* bytes;
  const uint32_t* meta;
  template<typename Type> const Type* get(unsigned index) const {
    return reinterpret_cast<const Type*>(bytes+meta[index]);
  }
  int32_t integer(unsigned index) const { int32_t x;std::memcpy(&x,meta+index,4);return x; }
};

void norm(const int8_t* x,int8_t* y,Params p,int index,int rows) {
  const int32_t* gamma=p.get<int32_t>(index),*beta=p.get<int32_t>(index+1);
  for (int row=0;row<rows;++row) {
    int64_t sum=0,ss=0;
    for (int i=0;i<D;++i) { int v=x[row*D+i];sum+=v;ss+=v*v; }
    uint64_t denom=isqrt(uint64_t(D*ss-sum*sum)+p.meta[index+2]);
    for (int i=0;i<D;++i) {
      int64_t numerator=(D*int64_t(x[row*D+i])-sum)*gamma[i];
      int64_t divisor=int64_t(denom)*(1<<16);
      y[row*D+i]=clip(floor_div(numerator+divisor/2,divisor)+beta[i]);
    }
  }
}
void dense(const int8_t* x,int8_t* y,Params p,int index,int rows,int k,int n,const int8_t* lut=nullptr) {
  project::linear<4>(x,p.get<int8_t>(index),p.get<int32_t>(index+1),p.get<int32_t>(index+2),
                     p.get<int32_t>(index+3),y,rows,k,n,lut);
}
void residual(const int8_t* a,const int8_t* b,int8_t* y,Params p,int index) {
  for (int i=0;i<T*D;++i)
    y[i]=clip(int64_t(scaled(a[i],p.integer(index),p.integer(index+1)))+
              scaled(b[i],p.integer(index+2),p.integer(index+3)));
}
void attention(Params p) {
  const uint32_t* exp=p.get<uint32_t>(26);
  for (int h=0;h<3;++h) {
    for (int token=0;token<T;++token) for (int d=0;d<64;++d) {
      queries[token*64+d]=qkv[token*576+h*64+d];
      keys[token*64+d]=qkv[token*576+192+h*64+d];
    }
    for (int c=0;c<T;++c) { biases[c]=0;multipliers[c]=p.integer(24);shifts[c]=p.integer(25); }
    project::linear<4>(queries,keys,biases,multipliers,shifts,scores,T,64,T);
    for (int row=0;row<T;++row) {
      int maximum=-128;
      for (int c=0;c<T;++c) if (scores[row*T+c]>maximum) maximum=scores[row*T+c];
      uint64_t sum=0;
      for (int c=0;c<T;++c) sum+=exp[maximum-int(scores[row*T+c])];
      for (int c=0;c<T;++c) {
        uint64_t e=exp[maximum-int(scores[row*T+c])];
        int q=int((e*255+sum/2)/sum);
        probs[row*PAD+c]=int8_t(q-128);
      }
      for (int c=T;c<PAD;++c) probs[row*PAD+c]=-128;
    }
    for (int c=0;c<64;++c) {
      int sum=0;
      for (int token=0;token<T;++token) {
        int8_t v=qkv[token*576+384+h*64+c]; values[c*PAD+token]=v;sum+=v;
      }
      for (int token=T;token<PAD;++token) values[c*PAD+token]=0;
      biases[c]=128*sum; multipliers[c]=p.integer(27);shifts[c]=p.integer(28);
    }
    project::linear<4>(probs,values,biases,multipliers,shifts,head_output,T,PAD,64);
    for (int token=0;token<T;++token) for (int c=0;c<64;++c)
      context_data[token*D+h*64+c]=head_output[token*64+c];
  }
}
void embedding(const int8_t* image,int8_t* y,Params p) {
  for (int py=0;py<14;++py) for (int px=0;px<14;++px)
    for (int c=0;c<3;++c) for (int dy=0;dy<16;++dy) for (int dx=0;dx<16;++dx)
      hidden[(py*14+px)*768+c*256+dy*16+dx]=image[(c*224+py*16+dy)*224+px*16+dx];
  dense(hidden,tmp,p,20,196,768,192);
  const int32_t* pos=p.get<int32_t>(24),*cls=p.get<int32_t>(25);
  for (int c=0;c<D;++c) y[c]=clip(int64_t(cls[c])+pos[c]);
  for (int i=D;i<T*D;++i)
    y[i]=clip(int64_t(scaled(tmp[i-D],p.integer(26),p.integer(27)))+pos[i]);
}
void block(const int8_t* x,int8_t* y,Params p) {
  norm(x,normalized,p,16,T);
  dense(normalized,qkv,p,20,T,D,576);
  attention(p);
  dense(context_data,tmp,p,32,T,D,D);
  residual(x,tmp,y,p,36);
  norm(y,normalized,p,40,T);
  dense(normalized,hidden,p,44,T,D,H,p.get<int8_t>(48));
  dense(hidden,tmp,p,52,T,H,D);
  residual(y,tmp,y,p,56);
}
uint32_t hash(const int8_t* x,unsigned n) {
  uint32_t h=2166136261u;
  for (unsigned i=0;i<n;++i) h=(h^uint8_t(x[i]))*16777619u;
  return h;
}
} // namespace

extern "C" {
volatile unsigned nodes_checked=0,blocks_checked=0,failed_node=0;
volatile uint32_t expected_stage_hash=0,actual_stage_hash=0;
}

static TfLiteStatus Prepare(TfLiteContext* context,TfLiteNode* node) {
  using namespace tflite::micro;
  if (node->inputs->size!=2 || node->outputs->size!=1) return kTfLiteError;
  auto* micro=GetMicroContext(context);
  auto* x=micro->AllocateTempInputTensor(node,0);
  auto* blob=micro->AllocateTempInputTensor(node,1);
  auto* y=micro->AllocateTempOutputTensor(node,0);
  bool valid=x && blob && y && x->type==kTfLiteInt8 && y->type==kTfLiteInt8 &&
    blob->type==kTfLiteUInt8 && blob->bytes>=256 && (reinterpret_cast<uintptr_t>(blob->data.uint8)&15)==0;
  if (valid) {
    const uint32_t* meta=reinterpret_cast<const uint32_t*>(blob->data.uint8);
    uint32_t sx,sy;std::memcpy(&sx,&x->params.scale,4);std::memcpy(&sy,&y->params.scale,4);
    valid=meta[0]<=2 && meta[3]==sx && meta[4]==sy && x->params.zero_point==0 && y->params.zero_point==0;
  }
  if(x) micro->DeallocateTempTfLiteTensor(x);
  if(blob) micro->DeallocateTempTfLiteTensor(blob);
  if(y) micro->DeallocateTempTfLiteTensor(y);
  return valid ? kTfLiteOk : kTfLiteError;
}
static TfLiteStatus Eval(TfLiteContext* context,TfLiteNode* node) {
  using namespace tflite::micro;
  const auto* x=GetEvalInput(context,node,0);
  const auto* blob=GetEvalInput(context,node,1);
  auto* y=GetEvalOutput(context,node,0);
  const uint8_t* bytes=GetTensorData<uint8_t>(blob);
  Params p{bytes,reinterpret_cast<const uint32_t*>(bytes)};
  const int8_t* input=GetTensorData<int8_t>(x);int8_t* output=GetTensorData<int8_t>(y);
  if (p.meta[0]==0) embedding(input,output,p);
  else if (p.meta[0]==1) { block(input,output,p);blocks_checked=blocks_checked+1; }
  else { norm(input,normalized,p,16,T);dense(normalized,output,p,20,1,D,1000); }
  uint32_t result=hash(output,GetTensorShape(y).FlatSize());
  if (result!=p.meta[2]) {
    failed_node=nodes_checked+1;expected_stage_hash=p.meta[2];actual_stage_hash=result;
    return kTfLiteError;
  }
  nodes_checked=nodes_checked+1;
  return kTfLiteOk;
}
tflite::TFLMRegistration RegisterDeitInteger() {
  return tflite::micro::RegisterOp(nullptr,Prepare,Eval);
}
