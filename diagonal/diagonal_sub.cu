#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <iostream>
#include "func_defs.h"
#include "definitions.h"
#include "diagonal_utils.h"
#include "diagonal_sub.h"

CREATE_DIAGONAL_OP_INNER(sub_diagonal_kernel,//shemem step
    T* local_input = input;
    T local_value = *values;
    #pragma unroll
    for(int64_t i = 0; i < flatten_dim; i++){
        shmem[current_thread] = local_input[i * len * len + len * current_thread + current_thread];
        local_input[i * len * len + len * current_thread + current_thread] = shmem[current_thread] - local_value;
    })

CREATE_DIAGONAL_OP_INNER(sub_diagonal_array_kernel, //shmem 2 x step
    T* local_input = input;
    T* local_value = values;
    shmem[len + current_thread] = local_value[current_thread];
    __syncthreads();
    #pragma unroll
    for(int64_t i = 0; i < flatten_dim; i++){
        shmem[current_thread] = local_input[i * len * len + len * current_thread + current_thread];
        local_input[i * len * len + len * current_thread + current_thread] = shmem[current_thread] - shmem[len+current_thread] ;
    }
)

CREATE_DIAGONAL_OP(sub)