#include <cstdint>
#include <cstring>
#include <cstdio>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "reference.h"
#ifdef __riscv
extern "C" const uint8_t full_model_data[],full_input_data[];
#else
alignas(16) static uint8_t full_model_data[8*1024*1024],full_input_data[3*224*224];
#endif
alignas(16) static uint8_t arena[512*1024];
extern tflite::TFLMRegistration RegisterDeitInteger();
extern "C" {
extern volatile unsigned nodes_checked,blocks_checked,failed_node;
volatile unsigned outputs_checked=0,arena_used=0,top1=0,failure_stage=0;
volatile int failed_index=-1;
}
int main() {
#ifndef __riscv
  FILE* model=std::fopen(MODEL_FILE,"rb"),*input=std::fopen(INPUT_FILE,"rb");
  if (!model || !input) return 1;
  if (std::fread(full_model_data,1,FULL_MODEL_BYTES,model)!=FULL_MODEL_BYTES ||
      std::fread(full_input_data,1,sizeof(full_input_data),input)!=sizeof(full_input_data)) return 1;
  std::fclose(model);std::fclose(input);
#endif
  tflite::MicroMutableOpResolver<3> resolver;
  auto reg=RegisterDeitInteger();
  for (const char* name: {"DEIT_EMBED_INT8","DEIT_BLOCK_INT8","DEIT_HEAD_INT8"})
    if (resolver.AddCustom(name,&reg)!=kTfLiteOk) return failure_stage=1;
  tflite::MicroInterpreter interpreter(tflite::GetModel(full_model_data),resolver,arena,sizeof(arena));
  if (interpreter.AllocateTensors()!=kTfLiteOk) return failure_stage=2;
  arena_used=interpreter.arena_used_bytes();
  std::memcpy(interpreter.input(0)->data.int8,full_input_data,3*224*224);
#ifdef __riscv
  __asm__ volatile(".globl inference_begin\ninference_begin:" ::: "memory");
#endif
  auto status=interpreter.Invoke();
#ifdef __riscv
  __asm__ volatile(".globl inference_end\ninference_end:" ::: "memory");
#endif
  if (status!=kTfLiteOk) {
#ifndef __riscv
    std::printf("FAIL node=%u stage=%u\n",failed_node,failure_stage);
#endif
    return failure_stage=3;
  }
  if (nodes_checked!=14 || blocks_checked!=12) return failure_stage=4;
  auto* result=interpreter.output(0)->data.int8;
  int best=0;
  for (int i=0;i<1000;++i) {
    if (result[i]!=expected_logits[i]) { failed_index=i;return failure_stage=5; }
    if (result[i]>result[best]) best=i;
  }
  outputs_checked=1000;top1=best;
#ifndef __riscv
  std::printf("PASS: full DeiT-Tiny: %u nodes, %u complete blocks, %u exact logits; top1=%u arena=%u\n",
      nodes_checked,blocks_checked,outputs_checked,top1,arena_used);
#endif
  return 0;
}
