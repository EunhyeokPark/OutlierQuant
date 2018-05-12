#include <TH/TH.h>
#include "ol_function.h"

int sort_inplace(THFloatTensor* input){
  int N = THFloatTensor_nElement(input);
  float* ptr = THFloatTensor_data(input);

  if(ptr == NULL)
    return 0;

  sort_host(N, ptr);
  THFloatTensor_resize1d(input, N);
  return 1;
}
