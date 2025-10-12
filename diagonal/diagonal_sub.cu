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
    if(threadIdx.x == 0)
        shmem[0] = local_value;
    __syncthreads();
    local_input[flatten_index * len * len + len * current_thread + current_thread] -= shmem[0];
)

CREATE_DIAGONAL_OP_INNER(sub_diagonal_array_kernel, //shmem 2 x step
    T* local_input = input;
    T* local_value = values;
    #pragma unroll
    for(uint64_t i = threadIdx.x; i < len; i += blockDim.x) {
        shmem[i] = local_value[i];
    }
    __syncthreads();
    local_input[flatten_index * len * len + len * current_thread + current_thread] -= shmem[current_thread] ;
)

CREATE_DIAGONAL_OP(sub)