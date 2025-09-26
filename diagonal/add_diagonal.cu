#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <iostream>

#ifndef THREAD_COUNT
    #define THREAD_COUNT 128
#endif

#ifndef BLOCK_COUNT
    #define BLOCK_COUNT 66
#endif

#define STEP_DIM BLOCK_COUNT * THREAD_COUNT

extern void check_tensor(const torch::Tensor&);

template<typename T>
__global__
void
add_diagonal_kernel(T* input, T* value, int64_t flatten_dim, int64_t step, int64_t len){
    int64_t thread_index = blockIdx.x * blockDim.x + threadIdx.x + THREAD_COUNT * BLOCK_COUNT * step;
    if (thread_index >= len) return;
    T* local_input = input;
    T local_value = *value;
    for(int64_t i = 0; i < flatten_dim; i++){
        local_input[i * len * len + len * thread_index + thread_index] += local_value;
    }
}

template<typename T>
__global__
void
add_diagonal_array_kernel(T* input, T* values, int64_t flatten_dim, int64_t step, int64_t len){
    int64_t thread_index = blockIdx.x * blockDim.x + threadIdx.x + THREAD_COUNT * BLOCK_COUNT * step;
    if (thread_index >= len) return;
    T* local_input = input;
    T* local_values = values;
    for(int64_t i = 0; i < flatten_dim; i++){
        local_input[i * len * len + len * thread_index + thread_index] += local_values[thread_index];
    }
}

torch::Tensor
add_diagonal(torch::Tensor input, torch::Tensor value){
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    check_tensor(input);
    auto input_shape = input.sizes();
    int64_t len = input.size(input.dim() - 1);
    input = input.view({-1, len, len});
    int64_t flatten_dim = input.size(0);
    input = input.contiguous();
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(input.device().index());
    if(!value.dim()){
        
        AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "add_diagonal", [&]{
            auto* input_ptr = input.data_ptr<scalar_t>();
            auto* scalar_val = value.data_ptr<scalar_t>();
            for(int64_t i = 0; i < len + STEP_DIM; i += STEP_DIM)
                add_diagonal_kernel<scalar_t><<<BLOCK_COUNT, THREAD_COUNT, 0, stream>>>(input_ptr, scalar_val, flatten_dim, i / STEP_DIM, len);
            return input;
        });
    }else{
        TORCH_CHECK(len==value.size(0),"Check dimension last two dimensions of the input, and value array length of value");
        AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "add_diagonal", [&]{
            auto* input_ptr = input.data_ptr<scalar_t>();
            auto* value_ptr = value.data_ptr<scalar_t>();
            for(int64_t i = 0; i < len + STEP_DIM; i += STEP_DIM)
                add_diagonal_array_kernel<scalar_t><<<BLOCK_COUNT, THREAD_COUNT, 0, stream>>>(input_ptr, value_ptr, flatten_dim, i / STEP_DIM, len);
            return input;
        });
    }
    cudaStreamSynchronize(stream);
    input = input.view(input_shape);
    input = input.contiguous();
    return input;
}