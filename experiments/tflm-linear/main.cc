#include <cstdint>
#include <cstring>
#ifndef __riscv
#include <cstdio>
#endif
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "model_data.h"

alignas(16) static uint8_t arena[512*1024];
extern "C" {
volatile int failure_stage = 0;
volatile int failed_index = -1;
volatile int expected_value = 0;
volatile int actual_value = 0;
volatile unsigned arena_used = 0;
volatile unsigned custom_calls = 0;
volatile unsigned outputs_checked = 0;
}
extern tflite::TFLMRegistration RegisterProjectFC();
extern const tflite::micro::OpDataFullyConnected* prepared_fc;
#ifdef USE_GELU_LUT
extern tflite::TFLMRegistration RegisterGeluLut();
extern "C" {
extern const int8_t* prepared_gelu_lut;
extern const int8_t* gelu_last_input;
extern volatile unsigned gelu_calls;
volatile unsigned lut_entries_checked=0;
volatile unsigned fc_outputs_checked=0;
void gelu_lut_apply(const int8_t*,int8_t*,unsigned,const int8_t*);
}
#endif

int main() {
  const auto* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) return failure_stage = 1;
#ifdef USE_GELU_LUT
  tflite::MicroMutableOpResolver<2> resolver;
#else
  tflite::MicroMutableOpResolver<1> resolver;
#endif
#ifdef USE_FUSION
  auto fc_registration=RegisterProjectFC();
  if (resolver.AddCustom("FC_GELU_LUT",&fc_registration)!=kTfLiteOk) return failure_stage=2;
#else
  if (resolver.AddFullyConnected(RegisterProjectFC()) != kTfLiteOk) return failure_stage = 2;
#endif
#ifdef USE_GELU_LUT
#ifndef USE_FUSION
  auto gelu_registration=RegisterGeluLut();
  if (resolver.AddCustom("GELU_LUT",&gelu_registration)!=kTfLiteOk) return failure_stage=11;
#endif
#endif
  tflite::MicroInterpreter interpreter(model, resolver, arena, sizeof(arena));
  if (interpreter.AllocateTensors() != kTfLiteOk) return failure_stage = 3;
  arena_used = interpreter.arena_used_bytes();
#ifdef USE_GELU_LUT
  int8_t all_inputs[256], all_outputs[256];
  for (int i=0; i<256; ++i) all_inputs[i]=static_cast<int8_t>(i-128);
  gelu_lut_apply(all_inputs,all_outputs,256,prepared_gelu_lut);
  for (int i=0; i<256; ++i) {
    if (all_outputs[i]!=expected_gelu_lut[i]) { failed_index=i; return failure_stage=12; }
  }
  lut_entries_checked=256;
#endif
  // Validate the exact scale-derived parameters used by TFLM preparation.
  auto* fc = prepared_fc;
  if (!fc->is_per_channel) return failure_stage = 4;
  for (int i=0; i<TEST_N; ++i) {
    if (fc->per_channel_output_multiplier[i] != expected_multipliers[i] ||
        fc->per_channel_output_shift[i] != expected_shifts[i]) {
      failed_index = i;
      return failure_stage = 5;
    }
  }
  auto* input = interpreter.input(0);
  if (input->type != kTfLiteInt8 || input->bytes != sizeof(input_data))
    return failure_stage = 6;
  std::memcpy(input->data.int8, input_data, sizeof(input_data));
#ifdef __riscv
  __asm__ volatile(".globl invoke_begin\ninvoke_begin:" ::: "memory");
#endif
  auto status = interpreter.Invoke();
#ifdef __riscv
  __asm__ volatile(".globl invoke_end\ninvoke_end:" ::: "memory");
#endif
  if (status != kTfLiteOk) return failure_stage = 7;
  auto* output = interpreter.output(0);
  if (output->type != kTfLiteInt8 || output->bytes != sizeof(expected_y))
    return failure_stage = 8;
  for (unsigned i=0; i<sizeof(expected_y); ++i) {
    if (output->data.int8[i] != expected_y[i]) {
      failed_index = i;
      expected_value = expected_y[i];
      actual_value = output->data.int8[i];
      return failure_stage = 9;
    }
  }
  outputs_checked = sizeof(expected_y);
#ifdef USE_GELU_LUT
#ifndef USE_FUSION
  // No later operator overwrites this two-operator graph's fc1 buffer.
  const auto* intermediate_data=gelu_last_input;
  for (unsigned i=0; i<sizeof(expected_fc_y); ++i) {
    if (intermediate_data[i]!=expected_fc_y[i]) { failed_index=i; return failure_stage=13; }
  }
  fc_outputs_checked=sizeof(expected_fc_y);
  if (gelu_calls!=1) return failure_stage=14;
#else
  fc_outputs_checked=0;
  if (gelu_calls!=0) return failure_stage=14;
#endif
#endif
#ifdef USE_CUSTOM_FC
  if (custom_calls != 1) return failure_stage = 10;
#else
  if (custom_calls != 0) return failure_stage = 10;
#endif
#ifndef __riscv
  std::printf("PASS: TFLM FullyConnected M=%d K=%d N=%d, checked=%u, arena=%u bytes\n",
              TEST_M, TEST_K, TEST_N, outputs_checked, arena_used);
#endif
  return 0;
}
