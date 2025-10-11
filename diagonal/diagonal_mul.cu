#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <iostream>
#include "func_defs.h"
#include "definitions.h"
#include "diagonal_utils.h"
#include "diagonal_mul.h"

CREATE_DIAGONAL_OP_INNER(mul_diagonal_kernel,
    T* local_input = input;
    T local_value = *values;
    #pragma unroll
    for(int64_t i = 0; i < flatten_dim; i++){
        local_input[i * len * len + len * thread_index + thread_index] *= local_value;
    })
CREATE_DIAGONAL_OP_INNER(mul_diagonal_array_kernel,
    T* local_input = input;
    T* local_value = values;
    #pragma unroll
    for(int64_t i = 0; i < flatten_dim; i++){
        local_input[i * len * len + len * thread_index + thread_index] *= local_value[thread_index];
    }
    )

CREATE_DIAGONAL_OP(mul)