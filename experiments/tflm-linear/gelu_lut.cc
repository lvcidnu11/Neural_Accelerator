// Offline scale-aware exact GELU, executed as an integer-only TFLM operator.
#include <cstdint>
#include <cstring>
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace {
struct Data {
  const int8_t* lut;
  float input_scale, output_scale;
  int32_t input_zero_point, output_zero_point;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  if (!buffer || length != 276 || std::memcmp(buffer,"GLU1",4) != 0) return nullptr;
  auto* data=static_cast<Data*>(context->AllocatePersistentBuffer(context,sizeof(Data)));
  if (!data) return nullptr;
  std::memcpy(&data->input_scale,buffer+4,4);
  std::memcpy(&data->output_scale,buffer+8,4);
  std::memcpy(&data->input_zero_point,buffer+12,4);
  std::memcpy(&data->output_zero_point,buffer+16,4);
  data->lut=reinterpret_cast<const int8_t*>(buffer+20);
  return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite::micro;
  if (!node->user_data || node->inputs->size != 1 || node->outputs->size != 1)
    return kTfLiteError;
  auto* data=static_cast<const Data*>(node->user_data);
  auto* micro=GetMicroContext(context);
  auto* input=micro->AllocateTempInputTensor(node,0);
  auto* output=micro->AllocateTempOutputTensor(node,0);
  bool valid=input && output && input->type==kTfLiteInt8 && output->type==kTfLiteInt8;
  if (valid) {
    valid=input->params.scale==data->input_scale && output->params.scale==data->output_scale &&
        data->input_scale>0 && data->output_scale>0 &&
        input->params.zero_point==data->input_zero_point &&
        output->params.zero_point==data->output_zero_point &&
        input->dims->size==output->dims->size;
    for (int i=0; valid && i<input->dims->size; ++i)
      valid=input->dims->data[i]==output->dims->data[i];
  }
  if (input) micro->DeallocateTempTfLiteTensor(input);
  if (output) micro->DeallocateTempTfLiteTensor(output);
  return valid ? kTfLiteOk : kTfLiteError;
}
}  // namespace

extern "C" {
const int8_t* prepared_gelu_lut=nullptr;
const int8_t* gelu_last_input=nullptr;
volatile unsigned gelu_calls=0;
}

// Also exercised outside the measured ROI with every possible INT8 input.
extern "C" __attribute__((noinline)) void gelu_lut_apply(
    const int8_t* input, int8_t* output, unsigned count, const int8_t* lut) {
  for (unsigned i=0; i<count; ++i)
    output[i]=lut[static_cast<int>(input[i])+128];
}

static TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
#ifdef __riscv
  __asm__ volatile(".globl gelu_begin\ngelu_begin:" ::: "memory");
#endif
  using namespace tflite::micro;
  const auto* data=static_cast<const Data*>(node->user_data);
  const auto* input=GetEvalInput(context,node,0);
  auto* output=GetEvalOutput(context,node,0);
  gelu_last_input=GetTensorData<int8_t>(input);
  gelu_calls=gelu_calls+1;
  gelu_lut_apply(GetTensorData<int8_t>(input),GetTensorData<int8_t>(output),
                 GetTensorShape(input).FlatSize(),data->lut);
#ifdef __riscv
  __asm__ volatile(".globl gelu_end\ngelu_end:" ::: "memory");
#endif
  return kTfLiteOk;
}

static TfLiteStatus PrepareAndExpose(TfLiteContext* context,TfLiteNode* node) {
  auto status=Prepare(context,node);
  if (status==kTfLiteOk) prepared_gelu_lut=static_cast<const Data*>(node->user_data)->lut;
  return status;
}

tflite::TFLMRegistration RegisterGeluLut() {
  return tflite::micro::RegisterOp(Init,PrepareAndExpose,Eval);
}
