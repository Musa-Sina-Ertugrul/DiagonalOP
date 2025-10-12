#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "definitions.h"
#include "diagonal_utils.h"
#include "diagonal_sum.h"

// Warp-level reduction using shuffle operations for better performance
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Specialized warp reduction for half precision (fp16)
template<>
__device__ __forceinline__ at::Half warp_reduce_sum<at::Half>(at::Half val) {
    half h_val = *reinterpret_cast<half*>(&val);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        h_val = __hadd(h_val, __shfl_down_sync(0xffffffff, h_val, offset));
    }
    return *reinterpret_cast<at::Half*>(&h_val);
}

// Specialized warp reduction for bfloat16
template<>
__device__ __forceinline__ at::BFloat16 warp_reduce_sum<at::BFloat16>(at::BFloat16 val) {
    nv_bfloat16 bf_val = *reinterpret_cast<nv_bfloat16*>(&val);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        bf_val = __hadd(bf_val, __shfl_down_sync(0xffffffff, bf_val, offset));
    }
    return *reinterpret_cast<at::BFloat16*>(&bf_val);
}

// Optimized kernel that handles multiple elements per thread and uses warp primitives
template<typename T>
__global__
void
sum_diagonal_kernel(T* input, T* output, int64_t flatten_dim, int64_t len) {
    extern __shared__ char shared_bytes[];
    T* sdata = reinterpret_cast<T*>(shared_bytes);

    int64_t batch_idx = blockIdx.x;
    int64_t tid = threadIdx.x;
    T* local_input = reinterpret_cast<T*>(input);

    // Each thread sums multiple diagonal elements
    T thread_sum = 0;
    #pragma unroll
    for (int64_t i = tid; i < len; i += blockDim.x) {
        thread_sum += local_input[batch_idx * len * len + i * len + i];
    }

    // Warp-level reduction
    thread_sum = warp_reduce_sum(thread_sum);

    // First thread in each warp writes to shared memory
    int lane = tid & 31;
    int warp_id = tid >> 5;

    if (lane == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (tid < 32) {
        T val = (tid < (blockDim.x >> 5)) ? sdata[tid] : T(0);
        val = warp_reduce_sum(val);

        if (tid == 0) {
            output[batch_idx] = val;
        }
    }
}


torch::Tensor
sum_diagonal(torch::Tensor input) {
    check_tensor(input);
    input = input.contiguous();
    auto input_shape = input.sizes();
    int64_t len = input.size(input.dim() - 1);
    input = input.view({-1, len, len});
    int64_t flatten_dim = input.size(0);
    torch::Tensor output = torch::empty({flatten_dim}, input.options());

    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(input.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "sum_diagonal", [&]{
        auto* input_ptr = input.data_ptr<scalar_t>();
        auto* output_ptr = output.data_ptr<scalar_t>();

        // Use more threads for better performance on large matrices
        size_t shared_mem_size = (THREAD_COUNT / 32) * sizeof(scalar_t); // One element per warp

        sum_diagonal_kernel<scalar_t><<<flatten_dim, THREAD_COUNT, shared_mem_size, stream>>>(input_ptr, output_ptr, flatten_dim, len);
    });

    auto final_shape = input_shape.slice(0, input_shape.size() - 2);
    cudaStreamSynchronize(stream);
    return output.view(final_shape);
}