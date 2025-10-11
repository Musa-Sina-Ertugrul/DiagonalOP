#include <torch/extension.h>
#include <cuda_runtime.h>
#include "diagonal_utils.h"
#include "definitions.h"

void check_tensor(const torch::Tensor& input){
    int64_t dim = input.dim();
    TORCH_CHECK(dim>1,"Tensor dim must be bigger than 1");
    TORCH_CHECK(input.size(dim-1)==input.size(dim-2),"Tensor last two dim must have same shape");
    TORCH_CHECK(input.is_cuda(),"Tensor must be at cuda device");
}