#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "definitions.h"
#include "diagonal_utils.h"
#include "diagonal_sum.h"

template<typename T>
__global__
void
sum_diagonal_kernel(const T* input, T* output, int64_t flatten_dim, int64_t len) {
    extern __shared__ char shared_bytes[];
    T* sdata = reinterpret_cast<T*>(shared_bytes);

    int64_t tid = threadIdx.x;
    int64_t batch_idx = blockIdx.x;
    int64_t i = threadIdx.x;

    if (i < len) {
        sdata[tid] = input[batch_idx * len * len + i * len + i];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    #pragma unroll
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch_idx] = sdata[0];
    }
}


torch::Tensor
sum_diagonal(torch::Tensor input) {
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    check_tensor(input);

    auto input_shape = input.sizes();
    int64_t len = input.size(input.dim() - 1);
    input = input.view({-1, len, len});
    int64_t flatten_dim = input.size(0);

    auto output_options = torch::TensorOptions().device(input.device()).dtype(input.dtype());
    torch::Tensor output = torch::zeros({flatten_dim}, output_options);

    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(input.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "sum_diagonal", [&]{
        auto* input_ptr = input.data_ptr<scalar_t>();
        auto* output_ptr = output.data_ptr<scalar_t>();

        size_t shared_mem_size = THREAD_COUNT * sizeof(scalar_t);

        sum_diagonal_kernel<scalar_t><<<flatten_dim, THREAD_COUNT, shared_mem_size, stream>>>(input_ptr, output_ptr, flatten_dim, len);
    });

    auto final_shape = input_shape.slice(0, input_shape.size() - 2);
    return output.view(final_shape);
}