#ifndef __FUNC_DEF__
#define __FUNC_DEF__

#define CREATE_DIAGONAL_OP_INNER(func_name,calculation) \
template<typename T> \
__global__ \
void \
func_name(T* input, T* values, uint64_t step, uint64_t len, uint64_t block_count, uint64_t flatten_dim){ \
    uint64_t thread_index = blockIdx.x * blockDim.x + threadIdx.x + THREAD_COUNT * block_count * step; \
    if (thread_index >= len) return; \
    do { \
        calculation \
    }while(0); \
}

#define CREATE_DIAGONAL_OP(func_name) \
torch::Tensor \
CONCAT(func_name,_diagonal)(torch::Tensor input, torch::Tensor value){ \
    input = input.contiguous(); \
    check_tensor(input); \
    auto input_shape = input.sizes(); \
    int64_t len = input.size(input.dim() - 1); \
    input = input.view({-1, len, len}); \
    uint64_t flatten_dim = input.size(0); \
    uint64_t block_count = (input.numel() / (flatten_dim*THREAD_COUNT)) + 1; \
    uint64_t step_dim = THREAD_COUNT * block_count; \
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(input.device().index()); \
    if(!value.dim()){ \
        AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "div_diagonal", [&]{ \
            auto* input_ptr = input.data_ptr<scalar_t>(); \
            auto* scalar_val = value.data_ptr<scalar_t>(); \
            for(uint64_t i = 0; i < len + step_dim; i += step_dim) \
                CONCAT(func_name,_diagonal_kernel)<scalar_t><<<block_count, THREAD_COUNT, 0, stream>>>(input_ptr, scalar_val, i / step_dim, len, block_count, flatten_dim); \
            return input; \
        }); \
    }else{ \
        TORCH_CHECK(len==value.size(0),"Check dimension last two dimensions of the input, and value array length of value"); \
        AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "div_diagonal", [&]{ \
            auto* input_ptr = input.data_ptr<scalar_t>(); \
            auto* value_ptr = value.data_ptr<scalar_t>(); \
            for(uint64_t i = 0; i < len + step_dim; i += step_dim) \
                CONCAT(func_name,_diagonal_array_kernel)<scalar_t><<<block_count, THREAD_COUNT, 0, stream>>>(input_ptr, value_ptr, i / step_dim, len, block_count, flatten_dim); \
            return input; \
        }); \
    } \
    input = input.view(input_shape); \
    input = input.contiguous(); \
    cudaStreamSynchronize(stream); \
    return input; \
}

#endif