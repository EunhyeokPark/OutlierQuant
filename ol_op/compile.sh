#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

mkdir ol_lib
cd src
echo "Compiling my kernel by nvcc..."
nvcc -O3 -c -o ../ol_lib/ol_function.o ol_function.cu -x cu -Xcompiler -fPIC  \
    -gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

cd ../
python build.py
