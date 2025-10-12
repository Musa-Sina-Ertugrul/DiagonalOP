#ifndef __FUNC_DEF__
#define __FUNC_DEF__

#define CREATE_DIAGONAL_OP_INNER(func_name,calculation) \
template<typename T> \
__global__ \
void \
func_name(T* input, T* values, uint64_t len, uint64_t flatten_dim){ \
    extern __shared__ char shared_bytes[];\
    T* shmem = reinterpret_cast<T*>(shared_bytes); \
    uint64_t current_thread = blockIdx.x * blockDim.x + threadIdx.x; \
    if (current_thread >= len) return; \
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
    int64_t len = input.size(-1); \
    input = input.view({-1, len, len}); \
    uint64_t flatten_dim = input.size(0); \
    uint64_t block_count = (input.numel() / (flatten_dim*THREAD_COUNT)) + 1; \
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(input.device().index()); \
    if(!value.dim()){ \
        AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "div_diagonal", [&]{ \
            auto* input_ptr = input.data_ptr<scalar_t>(); \
            auto* scalar_val = value.data_ptr<scalar_t>(); \
            CONCAT(func_name,_diagonal_kernel)<scalar_t><<<block_count, THREAD_COUNT, (len+1)*sizeof(scalar_t), stream>>>(input_ptr, scalar_val, len, flatten_dim); \
        }); \
    }else{ \
        TORCH_CHECK(len==value.size(0),"Check dimension last two dimensions of the input, and value array length of value"); \
        AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "div_diagonal", [&]{ \
            auto* input_ptr = input.data_ptr<scalar_t>(); \
            auto* value_ptr = value.data_ptr<scalar_t>(); \
            CONCAT(func_name,_diagonal_array_kernel)<scalar_t><<<block_count, THREAD_COUNT, (2*len+1)*sizeof(scalar_t), stream>>>(input_ptr, value_ptr, len, flatten_dim); \
        }); \
    } \
    cudaStreamSynchronize(stream); \
    input = input.view(input_shape); \
    input = input.contiguous(); \
    return input; \
}

#endif