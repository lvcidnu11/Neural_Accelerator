// Project adapter: preserve upstream Init/Prepare and provide an optional INT8
// Eval path. Unsupported configurations retain upstream behavior.
#include <cstring>
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#ifndef PROJECT_BLOCK
#define PROJECT_BLOCK 1
#endif
#ifdef USE_CUSTOM_FC
#include "linear_kernel.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "../qmac8-requant-spike/qmac8.h"
#include "../qmac8-requant-spike/requant8.h"
#endif

const tflite::micro::OpDataFullyConnected* prepared_fc = nullptr;
static tflite::TFLMRegistration upstream;
extern "C" volatile unsigned custom_calls;
#ifdef USE_FUSION
extern "C" const int8_t* prepared_gelu_lut;
#endif

static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
#ifdef USE_FUSION
  if (node->custom_initial_data_size!=276 ||
      std::memcmp(node->custom_initial_data,"GLU1",4)!=0) return kTfLiteError;
  auto* params=static_cast<TfLiteFullyConnectedParams*>(context->AllocatePersistentBuffer(context,sizeof(TfLiteFullyConnectedParams)));
  std::memset(params,0,sizeof(*params));
  params->activation=kTfLiteActNone;
  node->builtin_data=params;
  float input_scale,output_scale;
  const char* options=static_cast<const char*>(node->custom_initial_data);
  std::memcpy(&input_scale,options+4,4);std::memcpy(&output_scale,options+8,4);
  // Current graph deliberately uses the same fc1 and GELU output scale.
  if (!(input_scale>0) || input_scale!=output_scale) return kTfLiteError;
  prepared_gelu_lut=reinterpret_cast<const int8_t*>(options+20);
#endif
  auto status = upstream.prepare(context, node);
  if (status == kTfLiteOk)
    prepared_fc = static_cast<tflite::micro::OpDataFullyConnected*>(node->user_data);
  return status;
}

static TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
#ifdef __riscv
  __asm__ volatile(".globl operator_begin\noperator_begin:" ::: "memory");
#endif
  TfLiteStatus status;
#ifdef USE_CUSTOM_FC
  using namespace tflite::micro;
  const auto& data = *static_cast<const OpDataFullyConnected*>(node->user_data);
  const auto* x = GetEvalInput(context, node, kFullyConnectedInputTensor);
  const auto* w = GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const auto* bias = GetEvalInput(context, node, kFullyConnectedBiasTensor);
  auto* y = GetEvalOutput(context, node, kFullyConnectedOutputTensor);
  bool supported = x->type == kTfLiteInt8 && w->type == kTfLiteInt8 &&
      bias && bias->type == kTfLiteInt32 && y->type == kTfLiteInt8 &&
      data.is_per_channel && data.input_zero_point == 0 && data.filter_zero_point == 0 &&
      data.output_zero_point == 0 && data.output_activation_min == -128 &&
      data.output_activation_max == 127 && w->dims->size == 2;
  if (supported) {
    const int k = w->dims->data[1], n = w->dims->data[0];
    for (int c=0; c<n; ++c)
      supported = supported && data.per_channel_output_multiplier[c] >= 0 &&
          data.per_channel_output_shift[c] >= -31 && data.per_channel_output_shift[c] <= 30;
    // This experiment's exporter proves no INT32 overflow for this fixture.
    // General model integration must establish the same bound or use fallback.
    if (supported) {
      const int rows = GetTensorShape(x).FlatSize() / k;
      const int8_t* inputs = GetTensorData<int8_t>(x);
      const int8_t* weights = GetTensorData<int8_t>(w);
      const int32_t* biases = GetTensorData<int32_t>(bias);
      int8_t* outputs = GetTensorData<int8_t>(y);
      custom_calls = custom_calls + 1;
#if PROJECT_BLOCK>1 || defined(USE_FUSION) || defined(USE_PROJECT_KERNEL)
      const int8_t* lut=nullptr;
#ifdef USE_FUSION
      lut=prepared_gelu_lut;
#endif
      project::linear<PROJECT_BLOCK>(inputs,weights,biases,data.per_channel_output_multiplier,
          data.per_channel_output_shift,outputs,rows,k,n,lut);
#else
      for (int row=0; row<rows; ++row) {
        for (int c=0; c<n; ++c) {
          const int8_t *xp=inputs+row*k, *wp=weights+c*k;
          uint64_t acc = static_cast<uint64_t>(static_cast<int64_t>(biases[c]));
          int i=0;
          if (((reinterpret_cast<uintptr_t>(xp) | reinterpret_cast<uintptr_t>(wp)) & 7) == 0) {
            for (; k-i>=8; i+=8) {
              uint64_t a, b;
              std::memcpy(&a, __builtin_assume_aligned(xp+i,8), 8);
              std::memcpy(&b, __builtin_assume_aligned(wp+i,8), 8);
              acc = qmac8_bits(acc, a, b);
            }
          }
          uint32_t low=static_cast<uint32_t>(acc);
          int32_t scalar;
          std::memcpy(&scalar, &low, 4);
          for (; i<k; ++i) scalar += static_cast<int32_t>(xp[i])*wp[i];
#ifdef USE_QMAC_ONLY
          int32_t scaled=MultiplyByQuantizedMultiplier(scalar,
              data.per_channel_output_multiplier[c],data.per_channel_output_shift[c]);
          if (scaled>127) scaled=127;
          if (scaled< -128) scaled= -128;
          outputs[row*n+c]=static_cast<int8_t>(scaled);
#else
          auto params=requant8_params(data.per_channel_output_multiplier[c],
                                     data.per_channel_output_shift[c],0);
          outputs[row*n+c]=requant8_int8(scalar,params);
#endif
        }
      }
#endif
      status = kTfLiteOk;
    } else status = upstream.invoke(context,node);
  } else status = upstream.invoke(context,node);
#else
  status = upstream.invoke(context,node);
#endif
#ifdef __riscv
  __asm__ volatile(".globl operator_end\noperator_end:" ::: "memory");
#endif
  return status;
}

tflite::TFLMRegistration RegisterProjectFC() {
  upstream = tflite::Register_FULLY_CONNECTED();
  auto registration = upstream;
  registration.prepare = Prepare;
  registration.invoke = Eval;
  return registration;
}
