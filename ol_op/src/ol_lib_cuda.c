#include <THC/THC.h>
#include "ol_function.h"

#include <stdio.h>
// this symbol will be resolved automatically from PyTorch libs
extern THCState* state;

int sort_inplace_cuda(THCudaTensor* input){
  // TODO: exception handling (input, output check, continuous allocation, type check)

  int N = THCudaTensor_nElement(state, input);
  float* ptr = THCudaTensor_data(state, input);
  cudaStream_t stream = THCState_getCurrentStream(state);

  if(ptr == NULL)
    return 0;

  sort_device(N, ptr, stream);
  THCudaTensor_resize1d(state, input, N);
  return 1;
}


// outlier quantization with CUDA
//    out = x.new()
//    sort_lib.ol_quant_cuda(x, out, th_val, 16)
int ol_quant_cuda(THCudaTensor* input, THCudaTensor* output, float th_val, int N_lv){
  THCudaTensor_resizeAs(state, output, input);

  // TODO: exception handling (input, output check, continuous allocation, type check)

  int N = THCudaTensor_nElement(state, input);
  float* in = THCudaTensor_data(state, input);
  float* out = THCudaTensor_data(state, output);
  cudaStream_t stream = THCState_getCurrentStream(state);

  ol_quant(N, in, out, th_val, N_lv, stream);
  return 1;
}