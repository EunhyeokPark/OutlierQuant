#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include "ol_function.h"

#ifdef __cplusplus
extern "C" {
#endif

void sort_device(int N, float* ptr, cudaStream_t stream){
  thrust::device_ptr<float> dev = thrust::device_pointer_cast(ptr);
  thrust::sort(thrust::cuda::par.on(stream), dev, dev+N);
  CUDA_POST_KERNEL_CHECK;
}

void sort_host(int N, float* ptr){
  std::sort(ptr, ptr+N);
}

__global__ void ol_quant_kernel(int N, const float* in, float* out, float th_val, float diff){
  CUDA_KERNEL_LOOP(i, N)
  {    
    float tmp = in[i];    
    float sign = tmp >= 0 ? 1 : -1;
    tmp *= sign;
    tmp = tmp >= th_val ? tmp : roundf(tmp / diff) * diff;
    out[i] = tmp * sign;
  }
}

struct ol_functor: public thrust::unary_function<const float, float>{
  const float th_val;
  const float diff;

  ol_functor(float th_val, float diff):th_val(th_val), diff(diff){}


  __device__ float operator()(const float x){
    float sign = x >= 0 ? 1 : -1;
    float abs = x * sign;

    abs = abs >= th_val ? abs : roundf(abs/diff)*diff;
    return abs * sign;
  }

};

void ol_quant(int N, const float* in, float* out, float th_val, int N_lv_pos, cudaStream_t stream){ 
  float diff = th_val / N_lv_pos; 
  
  ol_quant_kernel <<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>
    (N, in, out, th_val, diff);
  CUDA_POST_KERNEL_CHECK;
  
  /*
  thrust::device_ptr<const float> in_d = thrust::device_pointer_cast(in);
  thrust::device_ptr<float> out_d = thrust::device_pointer_cast(out);

  thrust::transform(thrust::cuda::par.on(stream), in_d, in_d+N, out_d, ol_functor(th_val, diff) );
  */
}

#ifdef __cplusplus
}
#endif