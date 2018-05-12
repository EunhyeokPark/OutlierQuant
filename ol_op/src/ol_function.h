#ifndef _OL_FUNCTION_H_
#define _OL_FUNCTION_H_

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if(error != cudaSuccess) \
    { \
      fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( error ) ); \
      assert(0); \
    } \
  } while (0)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)      

// CUDA: use 512 threads per block
#define CUDA_NUM_THREADS 512

// CUDA: number of blocks for threads.
inline int CUDA_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

void sort_device(int N, float* ptr, cudaStream_t stream);
void sort_host(int N, float* ptr);

void ol_quant(int N, const float* in, float* out, float th_val, int N_lv_pos, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
